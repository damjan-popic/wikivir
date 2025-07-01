#!/usr/bin/env python3
"""run_bertopic.py – CPU‑friendly BERTopic pipeline for large Slovene corpus

This script embeds your cleaned documents with **paraphrase‑multilingual‑MPNet‑base‑v2**
(768‑dim, higher quality than MiniLM), stores them as a **memory‑mapped file** so
RAM never spikes, and then fits **BERTopic** with UMAP + HDBSCAN.

It’s designed for an 8‑core i7 with 16 GB RAM and can run unattended for a
few hours. Checkpoints mean you can resume halfway if SSH drops.

---------------------------------------------------------------------
Quick start (inside your virtualenv):

    pip install bertopic[visualization] sentence-transformers

    # docs.txt = one cleaned document per line (≤ 512 tokens each).
    python run_bertopic.py docs.txt --output slovene_topics

---------------------------------------------------------------------
Command‑line flags
------------------
--docs           Path to a text file with *one document per line* (UTF‑8).
--model          Sentence‑Transformer model (default: multilingual MPNet).
--batch-size     How many docs to embed at once (default 32 – safe on 16 GB).
--workers        CPU cores for embedding & BERTopic (default: os.cpu_count()).
--umap-dim       Dimensionality reduction target (default 5 – keeps RAM low).
--min-cluster    HDBSCAN min_cluster_size (default 25).
--output         Directory where checkpoints & final BERTopic model live.
--resume         If present, skips embedding chunks already on disk.

Outputs
-------
<output>/embeddings.dat        memory‑mapped float32 array (N × 768)
<output>/bertopic_model/       saved BERTopic model (topic_model.save())
<output>/topics.csv            doc‑to‑topic mapping for quick inspection

---------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import math
import mmap
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_docs(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="CPU‑friendly BERTopic runner")
    ap.add_argument("docs", help="Path to UTF‑8 file with one document per line")
    ap.add_argument("--output", default="bertopic_run", help="Output directory")
    ap.add_argument("--model", default="paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--umap-dim", type=int, default=5)
    ap.add_argument("--min-cluster", type=int, default=25)
    ap.add_argument("--resume", action="store_true", help="Skip already‑saved embedding chunks")
    args = ap.parse_args(argv)

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    # ---------------------------------------------------------------------
    # 1. Load / prepare documents
    # ---------------------------------------------------------------------
    docs_path = Path(args.docs)
    docs = read_docs(docs_path)
    n_docs = len(docs)
    print(f"Loaded {n_docs:,} documents from {docs_path}.")

    # ---------------------------------------------------------------------
    # 2. Create memmap file for embeddings (768‑dim float32)
    # ---------------------------------------------------------------------
    emb_path = out_dir / "embeddings.dat"
    emb_shape = (n_docs, 768)
    emb_map = np.memmap(emb_path, dtype="float32", mode="r+" if emb_path.exists() else "w+", shape=emb_shape)

    # ---------------------------------------------------------------------
    # 3. Sentence‑Transformer embeddings in batches
    # ---------------------------------------------------------------------
    model = SentenceTransformer(args.model, device="cpu")
    model.max_seq_length = 512  # truncate long docs automatically

    start = perf_counter()
    batch = args.batch_size
    for start_idx in range(0, n_docs, batch):
        end_idx = min(start_idx + batch, n_docs)
        if args.resume and not np.all(emb_map[start_idx:end_idx] == 0):
            # skip chunk already on disk (cheap non‑zero check)
            continue
        chunk_docs = docs[start_idx:end_idx]
        emb = model.encode(
            chunk_docs,
            batch_size=batch // 2,  # half of outer for memory safety
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")
        emb_map[start_idx:end_idx] = emb
        emb_map.flush()
        done = end_idx / n_docs * 100
        print(f"Embeddings {end_idx}/{n_docs} ({done:4.1f} %)", end="\r", flush=True)

    print(f"\n✓ Embeddings complete in {(perf_counter() - start)/60:.1f} min.")

    # ---------------------------------------------------------------------
    # 4. Fit BERTopic (UMAP + HDBSCAN) – this part uses RAM but fits in 16 GB
    # ---------------------------------------------------------------------
    topic_model = BERTopic(
        umap_model={
            "n_neighbors": 15,
            "n_components": args.umap_dim,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
        },
        hdbscan_model={
            "min_cluster_size": args.min_cluster,
            "min_samples": 5,
            "metric": "euclidean",
        },
        language="multilingual",
        calculate_probabilities=False,
        nr_topics="auto",
        verbose=True,
        nr_candidates=20,
        representation_model=KeyBERTInspired()
    )

    print("Fitting BERTopic … (this may take a couple of hours)")
    topics, probs = topic_model.fit_transform(docs, embeddings=emb_map)

    topic_model.save(out_dir / "bertopic_model")
    np.save(out_dir / "topics.npy", np.array(topics, dtype="int32"))
    print("✓ BERTopic model saved to", out_dir)

    # Quick per‑doc mapping CSV (id,topic)
    csv_path = out_dir / "topics.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        for idx, t in enumerate(topics):
            fh.write(f"{idx}\t{t}\n")
    print("Per‑doc topic index written to", csv_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted – exiting.")
        sys.exit(1)
