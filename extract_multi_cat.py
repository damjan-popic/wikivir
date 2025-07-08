#!/usr/bin/env python3.9
"""
extract_multi_cat.py  – list <doc> lines where categories="a, b, …" has > 1 entry

Output
------
multi_cat.tsv   (tab-separated)
    doc_index   title (or [no-title])   categories_as_found
"""
import xml.etree.ElementTree as ET, sys, csv

SRC = "normalized_corpus.xml"
OUT = "multi_cat.tsv"

out = open(OUT, "w", encoding="utf-8", newline="")
w   = csv.writer(out, delimiter="\t")
w.writerow(["doc_index", "title", "categories"])   # header

doc_i = 0
for ev, el in ET.iterparse(SRC, events=("end",)):
    if el.tag != "doc":
        continue
    doc_i += 1
    cats = el.get("categories", "")
    if cats:
        toks = [t.strip() for t in cats.split(",") if t.strip()]
        if len(toks) > 1:
            title = el.get("title") or "[no-title]"
            w.writerow([doc_i, title, ", ".join(toks)])
    el.clear()        # free memory

out.close()
print(f"✓ {doc_i:,} docs scanned – see {OUT}")
