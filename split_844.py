#!/usr/bin/env python3.9
# split_844.py – explode review_needed.txt into individual slugs
import re, pathlib, csv, itertools
SPLIT = "_kategorija"
rows = []
for raw in pathlib.Path("review_needed.txt").read_text(encoding="utf-8").splitlines():
    for part in [p for p in raw.split(SPLIT) if p]:
        rows.append([raw, part])          # parent, atom
csv.writer(open("exploded_844.tsv","w",encoding="utf-8",newline="")).writerows(
    [["composite","token"]]+rows)
print("exploded_844.tsv written –",len(rows),"rows")
