#!/usr/bin/env python3.9
"""
mapper.py – map unmapped tokens using:
   • header_map.csv  (existing rows)
   • authors.txt     (curated names)
   • cache.sqlite    (Wikisource authors + genres, built by ws_crawler.py)

Usage
-----
    python mapper.py unmapped.txt header_map.csv authors.txt cache.sqlite
Outputs
-------
    patch_header_map.csv   # append to header_map.csv
    review_needed.txt      # tokens that defaulted to "category"
"""
from __future__ import annotations
import csv, re, sqlite3, unicodedata, sys, pathlib
from tqdm import tqdm

# -------- helpers -----------------------------------------------------------
SLUG_RGX = re.compile(r"[^\w]+")

def slug(s: str) -> str:
    return SLUG_RGX.sub("_", unicodedata.normalize("NFD", s)
                        .encode("ascii", "ignore").decode().lower()).strip("_")

SHELF   = re.compile(r"^dela(_[a-z])?$|^dela[a-z]?$", re.I)
YEAR4   = re.compile(r"\b(\d{4})\b")
CENTURY = re.compile(r"^(\d{1,2})(?:_|\.?)(stoletje|stolje)", re.I)

def looks_title(tok: str) -> bool:
    words = tok.replace("_", " ").split()
    return len(words) >= 3 and words[0][0].isupper() and any(w[0].islower() for w in words[1:])

# -------- load local author set --------------------------------------------
def load_known_authors(header_csv: pathlib.Path, authors_txt: pathlib.Path) -> dict[str, str]:
    known = {}
    # from existing map
    with header_csv.open(newline="", encoding="utf-8") as fh:
        for raw, field, canon in csv.reader(fh):
            if field == "author":
                known[raw] = canon
    # from curated list
    for line in authors_txt.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            known[slug(name)] = name
    return known

# -------- cache look-ups ----------------------------------------------------
class Cache:
    def __init__(self, db_path: pathlib.Path):
        self.db = sqlite3.connect(db_path)
        self.db.execute("PRAGMA query_only = 1")

    def author(self, tok: str) -> str | None:
        row = self.db.execute("SELECT name FROM authors WHERE slug=?", (tok,)).fetchone()
        return row[0] if row else None

    def genre(self, tok: str) -> str | None:
        row = self.db.execute("SELECT name FROM genres WHERE slug=?", (tok,)).fetchone()
        return row[0] if row else None

# -------- classify one piece -----------------------------------------------
def classify_piece(tok: str, cache: Cache, known_auth: dict[str, str]):
    if SHELF.match(tok):
        return None  # throw away shelving codes

    if m := YEAR4.search(tok):
        return tok, "year", m.group(1)

    if m := CENTURY.match(tok):
        return tok, "century", m.group(1)

    if tok in known_auth:
        return tok, "author", known_auth[tok]

    if name := cache.author(tok):
        return tok, "author", name

    if g := cache.genre(tok):
        return tok, "genre", g

    if looks_title(tok):
        return tok, "title", tok.replace("_", " ").capitalize()

    return tok, "category", tok.replace("_", " ")

# -------- main --------------------------------------------------------------
def main():
    if len(sys.argv) != 5:
        sys.exit("Usage: mapper.py unmapped.txt header_map.csv authors.txt cache.sqlite")

    unmapped, header_csv, authors_txt, cache_db = map(pathlib.Path, sys.argv[1:])
    cache = Cache(cache_db)
    known_auth = load_known_authors(header_csv, authors_txt)

    patch_rows: set[tuple[str, str, str]] = set()
    review: list[str] = []

    for raw in tqdm(unmapped.read_text(encoding="utf-8").splitlines(), desc="Classifying"):
        for part in [p for p in raw.split("_kategorija") if p]:
            row = classify_piece(part, cache, known_auth)
            if not row:
                continue                        # skipped token
            raw_tok, field, val = row
            if field == "category":
                review.append(raw_tok)          # low-confidence → review
            else:
                patch_rows.add(row)             # high-confidence → patch

    # write patch
    with open("patch_header_map.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(sorted(patch_rows))
    print(f"✓ patch_header_map.csv written — {len(patch_rows):,} rows")

    # write review list
    if review:
        pathlib.Path("review_needed.txt").write_text(
            "\n".join(sorted(set(review))), encoding="utf-8")
        print(f"⚠  {len(set(review))} tokens need manual check → review_needed.txt")

if __name__ == "__main__":
    main()
