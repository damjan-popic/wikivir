#!/usr/bin/env python3
"""
List tokens that are still *not* present in header_map.csv
after all auto-seeding and manual curation.

Usage
-----
    python list_remaining_tokens.py annotated_corpus.xml  > remaining.txt
    # or multiple files / wildcards
    python list_remaining_tokens.py *.xml                 > remaining.txt
"""
import csv, re, unicodedata, sys
from pathlib import Path

MAP      = Path("header_map.csv")
DOC_RE   = re.compile(r"<doc\s+([^>]+)>", re.I)
ATTR_RE  = re.compile(r'(\w+)="([^"]+)"')

def slug(t: str) -> str:
    n = unicodedata.normalize("NFD", t).encode("ascii","ignore").decode()
    return re.sub(r"[^\w]+", "_", n.lower()).strip("_")

# 1) Load ALL mapped slugs (literal + regex ignored for counting simplicity)
known = set()
with MAP.open(newline="", encoding="utf-8") as fh:
    known = {row[0] for row in csv.reader(fh)}

def extract_tokens(line: str):
    m = DOC_RE.search(line)
    if not m:
        return []
    toks = []
    for attr, raw in ATTR_RE.findall(m.group(1)):
        val = raw.strip()
        if not val:
            continue
        if attr.lower() == "categories":
            toks += [t.strip().lower() for t in val.split(",") if t.strip()]
        else:
            toks.append(val.strip())
    return toks

remaining = set()
for fname in sys.argv[1:]:
    with open(fname, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for tok in extract_tokens(line):
                if slug(tok) not in known:
                    remaining.add(tok)

for tok in sorted(remaining, key=str.casefold):
    print(tok)

sys.stderr.write(f"Total remaining tokens: {len(remaining):,}\n")
