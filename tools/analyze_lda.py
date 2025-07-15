#!/usr/bin/env python3
"""
analyze_lda.py - Aggregate LDA topics and intersect with document metadata.

This script takes the outputs from the `run_lda.py` script and the original
metadata file to produce human-readable summary tables.

Outputs:
- `topic_summary.csv`: A table with one topic per row, showing its top keywords.
- `topics_by_author.csv`: A table showing the most frequent topics for each author.
- `topics_by_genre.csv`: A table showing the most frequent topics for each genre.
- `topics_by_century.csv`: A table showing the most frequent topics for each century.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_metadata(path: Path) -> pd.DataFrame:
    """Loads and pivots the metadata file for easy joining."""
    df = pd.read_csv(path, header=None, names=["doc_id", "field", "value"])
    # Filter out potential header rows if they exist
    df = df[~df['doc_id'].str.lower().isin(["doc_id", "raw", "id"])]
    
    # Pivot from long to wide format (doc_id, author, genre, ...)
    pivot_df = df.pivot(index="doc_id", columns="field", values="value").reset_index()
    return pivot_df

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and intersect LDA results with metadata."
    )
    parser.add_argument(
        "--lda-dir", required=True, type=Path,
        help="Path to the lda_{cpu|gpu} directory containing model outputs."
    )
    parser.add_argument(
        "--meta-file", required=True, type=Path,
        help="Path to the metadata CSV file (e.g., header_map.csv)."
    )
    parser.add_argument(
        "--output-dir", default=Path("analysis_results"), type=Path,
        help="Directory to save the analysis CSV files."
    )
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Data ---
    print("Loading data files...")
    try:
        topic_words_df = pd.read_csv(args.lda_dir / "topic_topwords.csv")
        doc_topics_df = pd.read_csv(
            args.lda_dir / "doc_topics.tsv",
            sep="\t",
            header=None,
            names=["doc_id", "topic_id", "weight"]
        )
        meta_df = load_metadata(args.meta_file)
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        return

    # --- 2. Aggregate Topic Keywords ---
    print("Aggregating topic keywords...")
    # Group by topic_id and join the 'token' strings
    topic_summary = (
        topic_words_df.groupby("topic_id")["token"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"token": "top_keywords"})
    )
    
    # Save the summary
    summary_path = args.output_dir / "topic_summary.csv"
    topic_summary.to_csv(summary_path, index=False)
    print(f"✓ Topic summary saved to {summary_path}")

    # --- 3. Join Topics with Metadata ---
    print("Joining topics with metadata...")
    
    # FIX: Ensure 'doc_id' columns are the same string type before merging
    doc_topics_df['doc_id'] = doc_topics_df['doc_id'].astype(str)
    meta_df['doc_id'] = meta_df['doc_id'].astype(str)

    # Merge document topics with their metadata
    merged_df = pd.merge(doc_topics_df, meta_df, on="doc_id", how="left")
    # Add the aggregated keywords for each topic
    full_df = pd.merge(merged_df, topic_summary, on="topic_id", how="left")
    
    # Fill any missing metadata with 'Unknown' for clean grouping
    for col in ["author", "genre", "century"]:
        if col not in full_df.columns:
            full_df[col] = "Unknown"
        else:
            full_df[col] = full_df[col].fillna("Unknown")

    # --- 4. Analyze Intersections and Export ---
    print("Analyzing intersections...")

    def analyze_and_export(grouping_col: str):
        """Helper function to group, count, and export results."""
        if grouping_col not in full_df.columns:
            print(f"Warning: Metadata column '{grouping_col}' not found. Skipping.")
            return

        # Count document occurrences for each class/topic pair
        analysis = (
            full_df.groupby([grouping_col, "topic_id", "top_keywords"])
            .size()
            .reset_index(name="doc_count")
        )
        # Sort for readability
        analysis = analysis.sort_values(
            by=[grouping_col, "doc_count"], ascending=[True, False]
        )
        
        # Save the result
        output_path = args.output_dir / f"topics_by_{grouping_col}.csv"
        analysis.to_csv(output_path, index=False)
        print(f"✓ Analysis for '{grouping_col}' saved to {output_path}")

    analyze_and_export("author")
    analyze_and_export("genre")
    analyze_and_export("century")

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
