#!/usr/bin/env python3.9
"""
normalize_categories.py
---------------------------------
• For rows where field == 'category':
    – strip Slovene diacritics
    – convert underscores to spaces
• Writes header_map_norm.csv.  Review it, then replace the old map:

    mv header_map_norm.csv header_map.csv
"""

import csv, unicodedata, pathlib, sys

SRC = pathlib.Path("header_map.csv")
DST = pathlib.Path("header_map_norm.csv")

if not SRC.exists():
    sys.exit("header_map.csv not found in current directory.")

def clean_category(text: str) -> str:
    sans = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()
    return sans.replace("_", " ")

rows, changed = [], 0
with SRC.open(encoding="utf-8") as fh:
    for row in csv.reader(fh):
        if len(row) != 3:
            print("⚠️ skipping malformed row:", row)
            continue
        token, field, val = row
        if field == "category":
            new_val = clean_category(val)
            if new_val != val:
                val = new_val
                changed += 1
        rows.append([token, field, val])

with DST.open("w", newline="", encoding="utf-8") as fh:
    csv.writer(fh).writerows(rows)

print(f"✓ wrote {DST}  —  normalised {changed} category values")
