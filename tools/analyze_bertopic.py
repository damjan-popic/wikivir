#!/usr/bin/env python3
"""analyze_bertopic.py – Generate BERTopic visualisations **and** export topic tables

HTML outputs (inside <run_dir>):
    ├─ heatmap_topics_per_year.html
    ├─ heatmap_topics_per_century.html
    ├─ barchart_topics_per_genre.html
    └─ barchart_topics_per_author.html

Tabular outputs:
    ├─ topic_keywords.csv      – one row per topic with size + top‑K lemmas
    └─ doc_topics.tsv          – doc_id \t topic_id
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from bertopic import BERTopic
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_docs(path: Path) -> tuple[List[str], List[str]]:
    """Return (doc_ids, lemma_lines).

    * If a tab is present, everything before the first tab is the doc_id.
    * Otherwise, doc_id is the running index and the whole line is lemmas.
    """
    ids: List[str] = []
    lemmas: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for idx, raw in enumerate(fh):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            if "\t" in raw:
                doc_id, lem = raw.split("\t", 1)
            else:
                doc_id, lem = str(idx), raw
            ids.append(doc_id)
            lemmas.append(lem)
    return ids, lemmas


def load_metadata(path: Path) -> Dict[str, Dict[str, str]]:
    """Row format: doc_id,field,value (header optional)."""
    meta: Dict[str, Dict[str, str]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].lower() in {"doc_id", "raw", "id"}:
                continue
            if len(row) < 3:
                continue
            doc_id, field, value = row[0].strip(), row[1].strip(), row[2].strip()
            meta[doc_id][field.lower()] = value
    return meta

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate BERTopic HTML + CSV exports")
    ap.add_argument("--run-dir", default="slovene_topics_full", help="Directory with bertopic_model/")
    ap.add_argument("--docs-file", default="docs.txt", help="Lemma lines (with or without doc_id prefix)")
    ap.add_argument("--meta-file", required=True, help="CSV/TSV mapping doc_id,field,value")
    ap.add_argument("--min-year", type=int, default=0, help="Skip docs before this year in year heatmap")
    ap.add_argument("--top-n", type=int, default=20, help="Top N topics to show in barcharts")
    ap.add_argument("--top-k", type=int, default=30, help="Top K lemmas to export per topic")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ids, docs = load_docs(Path(args.docs_file))

    print("Loading BERTopic artefacts …")
    topic_model: BERTopic = BERTopic.load(run_dir / "bertopic_model")
    topics = np.load(run_dir / "topics.npy")
    if len(ids) != len(topics):
        sys.exit(f"Mismatch: {len(ids)} docs vs {len(topics)} topics")

    print("Parsing metadata …")
    meta = load_metadata(Path(args.meta_file))

    # ---------------------------------------------------------------------
    # 1. Export doc→topic mapping (TSV)
    # ---------------------------------------------------------------------
    doc_topic_path = run_dir / "doc_topics.tsv"
    with doc_topic_path.open("w", encoding="utf-8") as fh:
        for doc_id, t in zip(ids, topics):
            fh.write(f"{doc_id}\t{int(t)}\n")
    print("✓ doc_topics.tsv written")

    # ---------------------------------------------------------------------
    # 2. Export topic keywords table
    # ---------------------------------------------------------------------
    print("Building topic_keywords.csv …")
    info_df = topic_model.get_topic_info()  # columns: Topic, Count, Name, etc.

    def join_top_words(topic_id: int, k: int = args.top_k) -> str:
        if topic_id == -1:
            return ""
        words = [w for w, _ in topic_model.get_topic(topic_id)[:k]]
        return " ".join(words)

    info_df["top_lemmas"] = info_df["Topic"].apply(lambda tid: join_top_words(int(tid)))
    info_df.rename(columns={"Topic": "topic_id", "Count": "size"}, inplace=True)
    info_df[["topic_id", "size", "top_lemmas"]].to_csv(run_dir / "topic_keywords.csv", index=False, encoding="utf-8")
    print("✓ topic_keywords.csv written")

    # ---------------------------------------------------------------------
    # 3. Prepare series for visualisations
    # ---------------------------------------------------------------------
    years, centuries, genres, authors = [], [], [], []
    for doc_id in tqdm(ids, desc="Metadata lookup"):
        m = meta.get(doc_id, {})
        y = int(m.get("year", 0)) if m.get("year", "0").isdigit() else 0
        c = int(m.get("century", 0)) if m.get("century", "0").isdigit() else 0
        years.append(y)
        centuries.append(c)
        genres.append(m.get("genre", "Neznano"))
        authors.append(m.get("author", "Neznan"))

    # Year heatmap
    print("Building heatmap_topics_per_year …")
    sel = [i for i, y in enumerate(years) if y >= args.min_year]
    if sel:
        to_df = topic_model.topics_over_time([docs[i] for i in sel], [years[i] for i in sel])
        fig = topic_model.visualize_topics_over_time(to_df)
        fig.write_html(run_dir / "heatmap_topics_per_year.html")
        print("✓ heatmap_topics_per_year.html saved")
    else:
        print("⚠ Ni dokumentov z letom ≥", args.min_year)

    # Century heatmap
    print("Building heatmap_topics_per_century …")
    sel = [i for i, c in enumerate(centuries) if c > 0]
    if sel:
        cent_years = [centuries[i] * 100 for i in sel]
        to_df = topic_model.topics_over_time([docs[i] for i in sel], cent_years)
        fig = topic_model.visualize_topics_over_time(to_df)
        fig.write_html(run_dir / "heatmap_topics_per_century.html")
        print("✓ heatmap_topics_per_century.html saved")
    else:
        print("⚠ Ni dokumentov z veljavno informacijo o stoletju")

    # Genre bar chart
    print("Building topics per genre bar chart …")
    df_gen = topic_model.topics_per_class(docs, classes=genres)
    fig = topic_model.visualize_topics_per_class(df_gen, top_n_topics=args.top_n)
    fig.write_html(run_dir / "barchart_topics_per_genre.html")
    print("✓ barchart_topics_per_genre.html saved")

    # Author bar chart
    print("Building topics per author bar chart …")
    df_aut = topic_model.topics_per_class(docs, classes=authors)
    fig = topic_model.visualize_topics_per_class(df_aut, top_n_topics=args.top_n)
    fig.write_html(run_dir / "barchart_topics_per_author.html")
    print("✓ barchart_topics_per_author.html saved")

    print("\nAll visualisations and CSVs saved inside", run_dir)


if __name__ == "__main__":
    main()
