#!/usr/bin/env python3
"""
metadata_stats.py – report coverage of metadata fields in a normalized corpus

Usage:
    python metadata_stats.py normalized_corpus.xml

Outputs to stdout:
  • Total number of documents
  • For each field: count & % of docs that have it
  • For “categories”: avg/median/min/max number of remaining tokens
"""
import sys
import xml.etree.ElementTree as ET
import statistics

def main(xml_path):
    fields = ["title", "author", "year", "century", "genre", "publication", "category"]
    counts = {f: 0 for f in fields}
    cat_counts = []
    total_docs = 0

    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != "doc":
            continue
        total_docs += 1

        # Check each field for presence
        for f in fields:
            val = el.get(f)
            if val and val.strip():
                counts[f] += 1

        # For categories, also record how many remain
        cats = el.get("categories", "")
        if cats.strip():
            toks = [c.strip() for c in cats.split(",") if c.strip()]
            cat_counts.append(len(toks))
        else:
            cat_counts.append(0)

        el.clear()

    # Print report
    print(f"Total documents: {total_docs}")
    print()
    for f in fields:
        cnt = counts[f]
        pct = (cnt / total_docs * 100) if total_docs else 0
        print(f"{f:12s} : {cnt:6d}  ({pct:5.1f}%)")

    print()
    print("Residual categories per doc:")
    if cat_counts:
        print(f"  Avg   : {statistics.mean(cat_counts):.2f}")
        print(f"  Median: {statistics.median(cat_counts)}")
        print(f"  Min   : {min(cat_counts)}")
        print(f"  Max   : {max(cat_counts)}")
    else:
        print("  (no data)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python metadata_stats.py normalized_corpus.xml")
    main(sys.argv[1])
