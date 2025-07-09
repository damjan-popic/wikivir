#!/usr/bin/env python3.9
"""
export_metadata_counts.py – dump frequency tables from normalized_corpus.xml

Creates these CSVs in ./stats/  (dir auto-created):

    docs_per_author.csv
    docs_per_year.csv
    docs_per_century.csv
    docs_per_genre.csv
    docs_per_publication.csv
    docs_per_category.csv   # counts each remaining category token

Usage:
    python export_metadata_counts.py normalized_corpus.xml
"""
from __future__ import annotations
import csv, os, statistics, xml.etree.ElementTree as ET, collections, sys, pathlib

FIELDS = ["author","year","century","genre","publication"]

def main(xml_path: str):
    counters = {f: collections.Counter() for f in FIELDS}
    cat_counter = collections.Counter()
    total_docs = 0

    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag != "doc":
            continue
        total_docs += 1

        for f in FIELDS:
            val = el.get(f)
            if val and val.strip():
                counters[f][val.strip()] += 1

        cats = el.get("categories","")
        if cats:
            for c in [x.strip() for x in cats.split(",") if x.strip()]:
                cat_counter[c] += 1

        el.clear()

    # ---------------------------------------------------------------- write CSVs
    outdir = pathlib.Path("stats")
    outdir.mkdir(exist_ok=True)
    for f, counter in counters.items():
        write_counter(outdir / f"docs_per_{f}.csv", counter)

    write_counter(outdir / "docs_per_category.csv", cat_counter)

    print(f"✓ wrote {len(FIELDS)+1} CSVs in {outdir}/  (total docs: {total_docs})")

def write_counter(path: pathlib.Path, counter: collections.Counter):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["key","doc_count"])
        for key, cnt in counter.most_common():
            w.writerow([key, cnt])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python export_metadata_counts.py normalized_corpus.xml")
    main(sys.argv[1])
