#!/usr/bin/env python
"""
run_bertopic_grid.py – GPU‑aware grid‑search wrapper **with built‑in metrics fallback**.

PATCH 2025‑07‑15 d (newline‑fix)
--------------------------------
* Removes unsupported `diversity=` arg for BERTopic ≤ 0.17.x.
* cuML reminder: GPU UMAP works only on Python 3.10 with `cuml‑cu12` wheels.
* Metrics fallback via `gensim` when BERTopic’s own helpers are absent.
* **Fixes stray newline in metrics writer.**

Quick sanity run (CPU UMAP):
```bash
python run_bertopic_grid.py \
  --docs docs.txt \
  --model intfloat/multilingual-e5-large \
  --umap-dim 30 --min-cluster 20 --min-samples 5 \
  --subset 5000 --fp16 --workers 8
```
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import multiprocessing as mp
import re
import string
from pathlib import Path
from typing import Any, List

import torch
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from rich.console import Console
from sentence_transformers import SentenceTransformer
from umap import UMAP  # overridden if --gpu-umap
import hdbscan

# gensim fallback for metrics ----------------------------------------------------
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

console = Console()

# --------------------------- metrics helpers ------------------------------------

def _simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [w for w in text.split() if len(w) > 2]


def _topic_words(topic_model: BERTopic, topn: int) -> List[List[str]]:
    topics: List[List[str]] = []
    for topic_id in topic_model.get_topics():
        if topic_id == -1:
            continue
        words = [w for w, _ in topic_model.get_topic(topic_id)[:topn]]
        topics.append(words)
    return topics


def _topic_diversity(topics: List[List[str]]) -> float:
    all_words = [w for topic in topics for w in topic]
    return len(set(all_words)) / max(len(all_words), 1)


def _coherence_npmi(
    topics: List[List[str]],
    tokenized_docs: List[List[str]],
    dictionary: Dictionary,
    workers: int,
) -> float:
    cm = CoherenceModel(
        topics=topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence="c_npmi",
        processes=max(workers, 1),
    )
    return cm.get_coherence()

# --------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid‑search BERTopic parameters on Slovene corpus (GPU‑aware)")
    p.add_argument("--docs", type=Path, required=True, help="Path to docs.txt (one doc per line)")
    p.add_argument("--model", required=True, help="Sentence‑Transformer model name or local path")
    p.add_argument("--umap-dim", type=int, default=30, help="UMAP n_components")
    p.add_argument("--min-cluster", type=int, default=20, help="HDBSCAN min_cluster_size")
    p.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min_samples")
    p.add_argument("--subset", type=int, default=None, help="Optional subset size for pilot run")
    p.add_argument("--batch-size", type=int, default=512, help="STS encode batch size (‑1 = auto)")
    p.add_argument("--output", type=Path, default=Path("outputs"), help="Directory to store results")
    p.add_argument("--fp16", action="store_true", help="Encode embeddings in float16 to save VRAM")
    p.add_argument("--gpu-umap", action="store_true", help="Use RAPIDS cuML UMAP if available")
    p.add_argument("--topn", type=int, default=10, help="Top‑N words per topic for metrics")
    p.add_argument("--workers", type=int, default=mp.cpu_count(), help="CPUs for gensim coherence")
    return p.parse_args()


def load_docs(path: Path, subset: int | None) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        docs = [line.split("\t", 1)[-1].strip() for line in f]
    return docs[:subset] if subset else docs


def build_umap(n_components: int, use_gpu: bool) -> Any:
    if use_gpu:
        try:
            from cuml.manifold import UMAP as GPUUMAP  # type: ignore
            console.print("[green]✓ Using cuML GPU UMAP[/green]")
            return GPUUMAP(n_components=n_components, n_neighbors=15, metric="cosine", random_state=42)
        except ImportError:
            console.print("[yellow]cuML not found – falling back to CPU UMAP[/yellow]")
    return UMAP(n_components=n_components, n_neighbors=15, metric="cosine", random_state=42)


def build_bertopic(args: argparse.Namespace, embed_model) -> BERTopic:
    umap_model = build_umap(args.umap_dim, args.gpu_umap)
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return BERTopic(
        embedding_model=embed_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        verbose=True,
    )


def evaluate_topics(topic_model: BERTopic, docs: list[str], topn: int, workers: int) -> dict[str, float | int]:
    console.print("[cyan]→ Computing metrics (gensim fallback)…")
    tokenized_docs = [_simple_tokenize(d) for d in docs]
    dictionary = Dictionary(tokenized_docs)
    topics = _topic_words(topic_model, topn)
    return {
        "diversity_score": _topic_diversity(topics),
        "c_npmi": _coherence_npmi(topics, tokenized_docs, dictionary, workers),
        "num_topics": len(topics),
    }


def auto_batch_size(vram_gb: int) -> int:
    return 512 if vram_gb >= 16 else 256


def main() -> None:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=run_dir / "run.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    console.rule(f"[bold blue]Loading documents from {args.docs}")
    docs = load_docs(args.docs, args.subset)
    console.print(f"Loaded {len(docs):,} documents")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Using device: {device}")

    console.rule(f"[bold blue]Loading embedding model: {args.model}")
    embed_model = SentenceTransformer(args.model, device=device)
    if args.fp16 and device == "cuda":
        embed_model = embed_model.half()
        console.print("[green]✓ FP16 enabled for embeddings[/green]")

    if args.batch_size == -1:
        vram_gb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3) if device == "cuda" else 0
        args.batch_size = auto_batch_size(vram_gb)
        console.print(f"Auto batch-size set to {args.batch_size}")

    console.rule("[bold blue]Building BERTopic model")
    topic_model = build_bertopic(args, embed_model)

    console.rule("[bold blue]Fitting model")
    console.print(f"[cyan]→ Encoding {len(docs):,} docs with batch_size={args.batch_size}…")
    embeddings = embed_model.encode(docs, batch_size=args.batch_size, show_progress_bar=True)
    _ = topic_model.fit_transform(docs, embeddings)

    console.rule("[bold blue]Evaluating topics")
    metrics = evaluate_topics(topic_model, docs, args.topn, args.workers)
    
    # Add run parameters to the metrics dictionary for logging
    run_params = vars(args)
    metrics["model"] = run_params.get("model")
    metrics["umap_dim"] = run_params.get("umap_dim")
    metrics["min_cluster"] = run_params.get("min_cluster")
    metrics["min_samples"] = run_params.get("min_samples")
    metrics["timestamp"] = timestamp

    # ensure JSON-serialisable (convert Path objects to str)
    clean_metrics = {k: (str(v) if isinstance(v, Path) else v) for k, v in metrics.items()}
    console.print(clean_metrics)

    console.rule("[bold blue]Saving artifacts")
    topic_model.save(run_dir / "bertopic_model", serialization="safetensors")
    
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as fh:
        # FIX: Use a single newline character
        fh.write(json.dumps(clean_metrics, ensure_ascii=False) + "\n")

    console.rule("[bold green]Done!")

if __name__ == "__main__":
    main()
