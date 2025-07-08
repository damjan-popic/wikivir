#!/usr/bin/env python3
"""
classify_wikisource_cats.py  –  conservative pre-seeder
-------------------------------------------------------
Reads categories_wikisource.txt and appends only
    • centuries  -> century
    • 2–4-word capitalised names -> author
to header_map.csv, skipping everything else.

Usage
-----
    python classify_wikisource_cats.py --dry   # show counts only
    python classify_wikisource_cats.py         # write new rows
"""
import argparse, csv, re, unicodedata
from pathlib import Path

CATS_FILE = Path("categories_wikisource.txt")
MAP_FILE  = Path("header_map.csv")

# ---------- helpers ---------------------------------------------------------

def slug(text: str) -> str:
    norm = unicodedata.normalize("NFD", text).encode("ascii","ignore").decode()
    return re.sub(r"[^\w]+", "_", norm.lower()).strip("_")

CENTURY_RE = re.compile(r"^\d{1,2}\.?\s*stoletje$", re.I)

def looks_like_author(name: str) -> bool:
    parts = name.split()
    if any(ch.isdigit() for ch in name) or not (2 <= len(parts) <= 4):
        return False
    return all(p[0].isupper() for p in parts)

# ---------- main ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="show counts; don’t modify header_map.csv")
    args = ap.parse_args()

    cats = CATS_FILE.read_text(encoding="utf-8").splitlines()

    keep_century, keep_author = [], []
    for cat in cats:
        if CENTURY_RE.match(cat):
            keep_century.append(cat)
        elif looks_like_author(cat):
            keep_author.append(cat)

    print(f"Centuries found : {len(keep_century):>5}")
    print(f"Authors   found : {len(keep_author):>5}")

    if args.dry:
        return

    # dedupe against existing rows
    existing = set()
    if MAP_FILE.exists():
        with MAP_FILE.open(newline="", encoding="utf-8") as fh:
            for raw, field, _ in csv.reader(fh):
                existing.add((raw, field))

    add_c = add_a = 0
    with MAP_FILE.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)

        for cat in keep_century:
            raw = slug(cat)
            if (raw, "century") not in existing:
                w.writerow([raw, "century", cat])
                add_c += 1

        for name in keep_author:
            raw = slug(name)
            if "_" in raw and (raw, "author") not in existing:
                w.writerow([raw, "author", name])
                add_a += 1

    print(f"✓ Appended {add_c} centuries, {add_a} authors → {MAP_FILE}")

if __name__ == "__main__":
    main()
