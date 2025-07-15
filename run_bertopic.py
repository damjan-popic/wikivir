#!/usr/bin/env python3
"""run_bertopic.py – GPU/CPU‑friendly BERTopic pipeline for a large Slovene corpus

Features
--------
* **Sentence‑Transformer embeddings** with automatic CUDA fallback (RTX 2000 Ada
  will be used if available).
* **Memory‑mapped** `(N × 768)` float32 array keeps RAM flat and supports
  `--resume` after drops.
* **Configurable** batch size, workers, UMAP dimension, and HDBSCAN parameters.
* **1‑click output** of the BERTopic model plus quick‑inspection CSVs.

Quick start
-----------
```bash
pip install bertopic[visualization] sentence-transformers umap-learn hdbscan
python run_bertopic.py docs.txt --output slovene_topics --batch-size 256
```"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_docs(path: Path) -> List[str]:
    """Load non‑empty lines from a UTF‑8 file."""
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return [line.rstrip("\n") for line in fh if line.strip()]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="BERTopic runner (GPU‑accelerated if available)")
    ap.add_argument("docs", help="Path to UTF‑8 file with one document per line")
    ap.add_argument("--output", default="bertopic_run", help="Output directory")
    ap.add_argument("--model", default="paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--batch-size", type=int, default=128, help="Embedding outer batch size")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="CPU cores")
    ap.add_argument("--umap-dim", type=int, default=5, help="UMAP components")
    ap.add_argument("--min-cluster", type=int, default=25, help="HDBSCAN min_cluster_size")
    ap.add_argument("--resume", action="store_true", help="Skip already‑saved embedding chunks")
    args = ap.parse_args(argv)

    out_dir = Path(args.output)
    ensure_dir(out_dir)

    # ---------------------------------------------------------------------
    # 1. Load documents
    # ---------------------------------------------------------------------
    docs_path = Path(args.docs)
    docs = read_docs(docs_path)
    n_docs = len(docs)
    print(f"Loaded {n_docs:,} documents from {docs_path}.")

    # ---------------------------------------------------------------------
    # 2. Prepare memory‑mapped embeddings array (70 MB for 22 k docs)
    # ---------------------------------------------------------------------
    emb_path = out_dir / "embeddings.dat"
    emb_shape = (n_docs, 768)
    emb_map = np.memmap(
        emb_path,
        dtype="float32",
        mode="r+" if emb_path.exists() else "w+",
        shape=emb_shape,
    )
    if not emb_path.exists():
        emb_map[:] = 0  # ensure zero‑init so resume check works
        emb_map.flush()

    # ---------------------------------------------------------------------
    # 3. Sentence‑Transformer embeddings
    # ---------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding device: {device}")
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = 512

    outer_batch = args.batch_size
    inner_batch = outer_batch if device == "cuda" else max(outer_batch // 2, 8)

    start = time.perf_counter()
    for start_idx in range(0, n_docs, outer_batch):
        end_idx = min(start_idx + outer_batch, n_docs)
        if args.resume and not np.all(emb_map[start_idx:end_idx] == 0):
            continue  # already done
        chunk_docs = docs[start_idx:end_idx]
        emb = model.encode(
            chunk_docs,
            batch_size=inner_batch,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype("float32")
        emb_map[start_idx:end_idx] = emb
        emb_map.flush()
        pct = end_idx / n_docs * 100
        print(f"Embeddings {end_idx}/{n_docs} ({pct:5.1f} %)", end="\r", flush=True)

    print(f"\n✓ Embeddings complete in {(time.perf_counter() - start)/60:.1f} min.")

    # ---------------------------------------------------------------------
    # 4. Fit BERTopic
    # ---------------------------------------------------------------------
    print("Fitting BERTopic … (this may take a while)")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=args.umap_dim,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdb_model = HDBSCAN(
        min_cluster_size=args.min_cluster,
        min_samples=5,
        metric="euclidean",
        core_dist_n_jobs=args.workers,
    )

    topic_model = BERTopic(
        embedding_model=model,
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        language="multilingual",
        calculate_probabilities=False,
        nr_topics="auto",
        verbose=True,
        representation_model=KeyBERTInspired(),
        low_memory=False,
    )

    topics, _ = topic_model.fit_transform(docs, embeddings=emb_map)

    # ---------------------------------------------------------------------
    # 5. Persist outputs
    # ---------------------------------------------------------------------
    model_dir = out_dir / "bertopic_model"
    topic_model.save(model_dir)
    print("✓ BERTopic model saved to", model_dir)

    topics_np = np.array(topics, dtype="int32")
    np.save(out_dir / "topics.npy", topics_np)

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
