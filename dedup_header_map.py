#!/usr/bin/env python3
"""
dedup_header_map.py  –  remove repeated (raw,field) rows, keep first
   python dedup_header_map.py header_map.csv > header_map_dedup.csv
"""
import csv, sys, pathlib

seen = set()
inp  = pathlib.Path(sys.argv[1])
outp = sys.stdout

with inp.open(newline="", encoding="utf-8") as fh, outp as oh:
    w = csv.writer(oh)
    for row in csv.reader(fh):
        if not row: continue
        key = tuple(row[:2])      # (raw, field)
        if key not in seen:
            seen.add(key)
            w.writerow(row)
