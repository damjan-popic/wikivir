#!/usr/bin/env python3
"""run_lda.py – LDA with optional *grid‑search coherence* to pick K

Usage patterns
==============
1. **Fixed K** (classic):
   ```bash
   python run_lda.py docs.txt --output lda_fixed --num-topics 60 --passes 15
   ```

2. **Automatic K via coherence grid**:
   ```bash
   python tools/run_lda.py docs.txt --output lda_grid --grid-search \
         --min-k 20 --max-k 200 --step 20 --grid-passes 5 --passes 15
   ```
   * grid‑passes = light training per K (fast)
   * passes      = final full‑quality training at the best K

The grid search runs **CPU‑only** (gensim) to keep dependencies simple; the
final model can still train on GPU if `--gpu` and a working RAPIDS stack are
available.

Outputs
-------
* `k_coherence.csv`  – table of K vs coherence (if grid enabled)
* `lda_{cpu|gpu}/…`  – model, `topic_topwords.csv`, `doc_topics.tsv`
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

# --------------------------- shared helpers ------------------------------

def iter_docs(path: Path) -> Iterable[List[str]]:
    """Yield tokenised documents from docs.txt."""
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                _, text = line.split("\t", 1)
            else:
                text = line
            yield text.split()


def load_doc_ids(path: Path) -> List[str]:
    """Load document IDs from a docs.txt file."""
    ids: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            ids.append(line.split("\t", 1)[0] if "\t" in line else str(idx))
    return ids

# --------------------------- CPU training & export -------------------------------

def train_lda_gensim(dictionary, corpus, k: int, passes: int, workers: int, chunksize: int):
    """Train a single gensim LdaMulticore model."""
    import gensim
    return gensim.models.LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=passes,
        workers=workers,
        chunksize=chunksize,
        eval_every=None,
        random_state=42,
    )

def _export_topwords_gensim(model, csv_path: Path, top_k: int) -> None:
    """Export top K words for each topic to a CSV file."""
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("topic_id,weight,token\n")
        for tid in range(model.num_topics):
            for token, weight in model.show_topic(tid, topn=top_k):
                fh.write(f"{tid},{weight:.5f},{token}\n")
    print("✓", csv_path.name)


def _export_doc_topics_gensim(model, corpus, docs_path: Path, tsv_path: Path) -> None:
    """Export the dominant topic for each document to a TSV file."""
    doc_ids = load_doc_ids(docs_path)
    with tsv_path.open("w", encoding="utf-8") as fh:
        for doc_id, bow in tqdm(zip(doc_ids, corpus), total=len(doc_ids), desc="Inferring doc topics (CPU)"):
            tid, prob = max(model.get_document_topics(bow, minimum_probability=0.0), key=lambda x: x[1])
            fh.write(f"{doc_id}\t{tid}\t{prob:.4f}\n")
    print("✓", tsv_path.name)

# ------------------------- coherence utility ----------------------------

def coherence_cv(model, texts, dictionary, corpus):
    """Calculate c_v coherence for a gensim model."""
    from gensim.models.coherencemodel import CoherenceModel
    cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, corpus=corpus, coherence="c_v")
    return cm.get_coherence()

# ------------------------- GPU helper ---------------------------------

def run_gpu_lda(docs_path: Path, out_dir: Path, num_topics: int, max_iter: int, top_k: int) -> bool:
    """Train cuML LDA and export artefacts. Returns True on success."""
    try:
        from cuml.feature_extraction.text import TfidfVectorizer
        from cuml.topic_model import LDATopicModel
        import cudf, joblib
    except Exception as e:
        print("⚠ GPU LDA unavailable –", e)
        print("  Tip: pip install cudf-cu12 cuml-cu12 (and cupy-cuda12x)")
        return False

    # ---- TF-IDF with same pruning as CPU ----
    docs_flat = [" ".join(t) for t in iter_docs(docs_path)]
    gdf = cudf.DataFrame({"text": docs_flat})
    
    # FIX #1: Create vectorizer object first to get vocab later
    tfidf = TfidfVectorizer(lowercase=False, min_df=5, max_df=0.5)
    X = tfidf.fit_transform(gdf["text"])
    vocab = tfidf.get_feature_names_out()

    # ---- Train LDA ----
    print("[GPU] Training cuML LDA …")
    lda = LDATopicModel(n_components=num_topics, max_iter=max_iter, random_state=42)
    lda.fit(X)
    joblib.dump(lda, out_dir / "lda_model_gpu.joblib")
    print("✓ GPU LDA model saved")

    # ---- Export top-K words per topic ----
    topic_word = lda.components_
    top_csv = out_dir / "topic_topwords.csv"
    with top_csv.open("w", encoding="utf-8") as fh:
        fh.write("topic_id,weight,token\n")
        for tid in range(num_topics):
            row = topic_word[tid].to_pandas().values
            for idx in row.argsort()[::-1][:top_k]:
                fh.write(f"{tid},{row[idx]:.6f},{vocab[idx]}\n")
    print("✓", top_csv.name)

    # ---- Export dominant topic per doc ----
    doc_ids = load_doc_ids(docs_path)
    probs = lda.transform(X)
    tsv_path = out_dir / "doc_topics.tsv"
    with tsv_path.open("w", encoding="utf-8") as fh:
        # FIX #2: Convert to pandas for safe and efficient iteration
        probs_pd = probs.to_pandas()
        dominant_topics = probs_pd.idxmax(axis="columns")
        max_probs = probs_pd.max(axis="columns")
        
        for doc_id, tid, prob in zip(doc_ids, dominant_topics, max_probs):
            fh.write(f"{doc_id}\t{tid}\t{prob:.4f}\n")
    print("✓", tsv_path.name)
    return True

# ------------------------- Main pipeline --------------------------------

def main():
    p = argparse.ArgumentParser("LDA with optional K grid search")
    p.add_argument("docs", help="Path to docs.txt")
    p.add_argument("--output", default="lda_run", help="Output directory")
    p.add_argument("--num-topics", type=int, default=50, help="K for fixed run or final model")
    p.add_argument("--passes", type=int, default=10, help="CPU passes / GPU max_iter")
    p.add_argument("--chunksize", type=int, default=2000, help="gensim chunksize")
    p.add_argument("--workers", type=int, default=mp.cpu_count(), help="gensim workers")
    p.add_argument("--top-k", type=int, default=30, help="Top K words to export")
    p.add_argument("--gpu", action="store_true", help="Use GPU for final model if RAPIDS present")

    # grid-search flags
    p.add_argument("--grid-search", action="store_true", help="Scan K range with coherence")
    p.add_argument("--min-k", type=int, default=20)
    p.add_argument("--max-k", type=int, default=200)
    p.add_argument("--step", type=int, default=20)
    p.add_argument("--grid-passes", type=int, default=5, help="Light passes for grid search models")
    args = p.parse_args()

    docs_path = Path(args.docs)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build dictionary + corpus once ---
    print("[prep] Building dictionary …")
    import gensim.corpora as corpora
    texts = list(iter_docs(docs_path)) # Load all texts for coherence calculation
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    corpus_path = out_dir / "bow.mm"
    print("[prep] Serialising corpus …")
    corpora.MmCorpus.serialize(str(corpus_path), (dictionary.doc2bow(t) for t in texts))
    corpus = corpora.MmCorpus(str(corpus_path))

    best_k = args.num_topics

    # --- Optional grid search for best K ---
    if args.grid_search:
        from csv import writer
        print(f"[grid] Searching K from {args.min_k} to {args.max_k} (step {args.step})")
        rows = []
        k_range = range(args.min_k, args.max_k + 1, args.step)
        for k in tqdm(k_range, desc="Grid Search"):
            model = train_lda_gensim(dictionary, corpus, k, args.grid_passes, args.workers, args.chunksize)
            coh = coherence_cv(model, texts, dictionary, corpus)
            rows.append((k, coh))
        
        with (out_dir / "k_coherence.csv").open("w", newline="", encoding="utf-8") as f:
            w = writer(f)
            w.writerow(["k", "c_v"])
            w.writerows(rows)
        
        best_k, best_cv = max(rows, key=lambda x: x[1])
        print(f"[grid] Best K={best_k} (c_v={best_cv:.3f}) found. Training final model...")

    # --- Final model (GPU or CPU) ---
    final_model_trained = False
    if args.gpu:
        gpu_dir = out_dir / "lda_gpu"
        gpu_dir.mkdir(exist_ok=True)
        if run_gpu_lda(docs_path, gpu_dir, best_k, args.passes, args.top_k):
            print(f"[final] GPU model (K={best_k}) saved in {gpu_dir}")
            final_model_trained = True

    if not final_model_trained:
        print(f"[final] Training CPU model (K={best_k})...")
        cpu_dir = out_dir / "lda_cpu"
        cpu_dir.mkdir(exist_ok=True)
        model = train_lda_gensim(dictionary, corpus, best_k, args.passes, args.workers, args.chunksize)
        model.save(str(cpu_dir / "lda_model"))
        _export_topwords_gensim(model, cpu_dir / "topic_topwords.csv", args.top_k)
        _export_doc_topics_gensim(model, corpus, docs_path, cpu_dir / "doc_topics.tsv")
        print(f"[final] CPU model (K={best_k}) saved in {cpu_dir}")

if __name__ == "__main__":
    main()
