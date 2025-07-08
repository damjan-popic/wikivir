#!/usr/bin/env python3
"""
validate_header_map.py  –  sanity-check a big header_map.csv

Run:
    python validate_header_map.py header_map.csv normalized_corpus.xml
"""

import csv, re, sys, unicodedata
from collections import Counter
from pathlib import Path

FIELDS = {"author","year","century","genre","category","title","language","publication"}

def slug_ok(s):
    return bool(re.match(r"^[a-z0-9_]+$", s))

def load_map(path):
    seen = Counter()
    errors = []
    with path.open(newline="",encoding="utf-8") as fh:
        for lineno,(raw,field,val,*rest) in enumerate(csv.reader(fh),1):
            # column count
            if rest: errors.append(f"{lineno}: Too many columns")
            # allowed field
            if field not in FIELDS:
                errors.append(f"{lineno}: bad field '{field}'")
            # value non-empty
            if not val.strip():
                errors.append(f"{lineno}: empty value")
            # slug vs regex
            if raw.startswith("^"):
                try: re.compile(raw)
                except re.error as e:
                    errors.append(f"{lineno}: bad regex – {e}")
            elif not slug_ok(raw):
                errors.append(f"{lineno}: raw not slug")
            seen[(raw,field)] += 1
    # duplicates
    for (raw,field),n in seen.items():
        if n>1:
            errors.append(f"dup ({raw},{field}) ×{n}")
    return errors

def slug(t):
    n = unicodedata.normalize("NFD",t).encode("ascii","ignore").decode()
    return re.sub(r"[\\W]+","_",n.lower()).strip("_")

def corpus_unmapped(corpus,map_path):
    known=set()
    with map_path.open(newline="",encoding="utf-8") as fh:
        for raw,field,_ in csv.reader(fh):
            known.add(raw)
    unknown=set()
    DOC=re.compile(r"<doc\\s+([^>]+)>",re.I)
    ATR=re.compile(r'(\\w+)="([^"]+)"')
    with open(corpus,encoding="utf-8",errors="ignore") as fh:
        for line in fh:
            m=DOC.search(line)
            if not m: continue
            for attr,val in ATR.findall(m.group(1)):
                if attr.lower()=="categories":
                    for t in val.split(","):
                        s=slug(t.strip())
                        if s and s not in known: unknown.add(s)
                else:
                    s=slug(val.strip())
                    if s and s not in known: unknown.add(s)
    return unknown

def main():
    if len(sys.argv)<3:
        sys.exit("Usage: validate_header_map.py header_map.csv normalized_corpus.xml")
    csv_path=Path(sys.argv[1]); corpus=Path(sys.argv[2])

    errs=load_map(csv_path)
    if errs:
        print("❌ Issues found:")
        for e in errs[:50]:
            print("  ",e)
        if len(errs)>50: print("  …and",len(errs)-50,"more")
    else:
        print("✓ CSV structure looks good.")

    unk=corpus_unmapped(corpus,csv_path)
    print(f"Corpus slugs still unmapped: {len(unk):,}")
    if unk:
        print("Sample:",", ".join(sorted(list(unk))[:20]))

if __name__=="__main__":
    main()
