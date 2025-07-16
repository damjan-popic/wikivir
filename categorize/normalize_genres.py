#!/usr/bin/env python3.9
"""
normalize_genres.py – ASCII + spaces in genre value column only.
"""
import csv, unicodedata, pathlib, sys

SRC = pathlib.Path("header_map.csv")
DST = pathlib.Path("header_map_norm.csv")

def clean(text: str) -> str:
    no_diac = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode()
    return no_diac.replace("_", " ")

rows, changed = [], 0
for r in csv.reader(SRC.open(encoding="utf-8")):
    if len(r) != 3:
        print("⚠ malformed row:", r); continue
    token, field, val = r
    if field == "genre":
        new = clean(val)
        if new != val:
            val = new
            changed += 1
    rows.append([token, field, val])

csv.writer(DST.open("w", newline="", encoding="utf-8")).writerows(rows)
print(f"✓ {changed} genre values normalised → {DST}")
