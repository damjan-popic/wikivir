#!/usr/bin/env python3.9
"""ws_fingerprint_enrich.py – enrich header_map.csv by fingerprint‐matching
Slovene Wikisource pages.

* For every <doc> in a **clean** corpus, compute a SHA‑1 fingerprint of the
  first 300 normalised characters.
* Look up that fingerprint in a local SQLite cache (fp_cache). If unknown,
  try to locate the page on sl.wikisource.org via:
      1. exact title + optional author incategory
      2. fallback full‑text search on the first 20 tokens
* If a page is found, pull its categories/templates/pageprops once and cache.
* Classify each category into publication / genre / category via regex hints.
* Emit **wikisource_patch.csv** with rows (token, field, value) that do not
  yet exist in your current header_map.csv.
* Safe to interrupt – cache.sqlite prevents re‑crawling.

Run overnight:
    python ws_fingerprint_enrich.py \
        --xml normalized_corpus.xml \
        --map header_map.csv \
        --out wikisource_patch.csv
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import re
import sqlite3
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional

import requests
from tqdm import tqdm

# -----------------------------------------------------------------------------
API = "https://sl.wikisource.org/w/api.php"
HEAD = {"User-Agent": "wikivir-fp-meta/0.2 (damjan@example.com)"}
RATE = 1.0  # sec between API calls (polite)

SLUG_RGX = re.compile(r"[^\w]+")
TITLE_CLEAN = re.compile(r"\s+")
FULLTEXT_LEN = 20  # tokens when doing fallback full‑text search

# hints for quick cat→field mapping (case‑insensitive, mods allowed)
PUB_HINT = re.compile(r"vestnik|glas|list|casopis|tednik|dnevnik", re.I)
GENRE_HINT = re.compile(r"poezij|pesm|roman|novel|drame|drama|humoresk|potopis|povest", re.I)

# ----------------------------------------------------------------------------- helpers

def slug(text: str) -> str:
    return SLUG_RGX.sub("_", unicodedata.normalize("NFD", text)
                        .encode("ascii", "ignore").decode().lower()).strip("_")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    return TITLE_CLEAN.sub(" ", text).strip()

def fingerprint(text: str, n_chars: int = 300) -> str:
    """Return SHA‑1 of first n normalised chars."""
    return hashlib.sha1(normalize(text)[:n_chars].encode()).hexdigest()

# ----------------------------------------------------------------------------- cache

def open_cache(path: Path = Path("cache.sqlite")) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS fp_cache(
            fp      TEXT PRIMARY KEY,
            pageid  INTEGER,    -- NULL pending, -1 not found
            ts      REAL
        );
        CREATE TABLE IF NOT EXISTS meta_cache(
            pageid  INTEGER PRIMARY KEY,
            categories TEXT
        );
        """
    )
    return db

# ----------------------------------------------------------------------------- API helpers

def api_get(sess: requests.Session, **params):
    time.sleep(RATE)
    r = sess.get(API, params={**params, "format": "json"}, headers=HEAD, timeout=20)
    r.raise_for_status()
    return r.json()

def find_page(title: str, author: Optional[str], fp: str, sess: requests.Session, db: sqlite3.Connection) -> Optional[int]:
    row = db.execute("SELECT pageid FROM fp_cache WHERE fp=?", (fp,)).fetchone()
    if row:  # cached result (None or -1 or real id)
        return row[0] if row[0] and row[0] > 0 else None

    # first try intitle search
    srsearch = f'intitle:"{title}"'
    if author:
        srsearch += f' incategory:"{author.split()[0]}"'
    try:
        hits = api_get(sess, action="query", list="search", srsearch=srsearch, srlimit=1)["query"]["search"]
        pageid = hits[0]["pageid"] if hits else None
    except (requests.RequestException, KeyError):
        hits = []
        pageid = None

    # fallback: fulltext search on first N tokens
    if pageid is None:
        tokens = normalize(title).split()[:FULLTEXT_LEN]
        if tokens:
            q = " ".join(tokens)
            try:
                hits = api_get(sess, action="query", list="search", srsearch=q, srlimit=1)["query"]["search"]
                pageid = hits[0]["pageid"] if hits else None
            except (requests.RequestException, KeyError):
                pageid = None

    db.execute("INSERT INTO fp_cache(fp,pageid,ts) VALUES(?,?,?)", (fp, pageid or -1, time.time()))
    db.commit()
    return pageid

def pull_categories(pageid: int, sess: requests.Session, db: sqlite3.Connection) -> List[str]:
    row = db.execute("SELECT categories FROM meta_cache WHERE pageid=?", (pageid,)).fetchone()
    if row:
        return row[0].split("|") if row[0] else []

    try:
        data = api_get(sess, action="query", pageids=pageid, prop="categories", cllimit="max")
        page = next(iter(data["query"]["pages"].values()))
        cats = [c["title"].split(":", 1)[-1] for c in page.get("categories", [])]
    except (requests.RequestException, KeyError, StopIteration):
        cats = []

    db.execute("INSERT OR REPLACE INTO meta_cache(pageid,categories) VALUES(?,?)", (pageid, "|".join(cats)))
    db.commit()
    return cats

# ----------------------------------------------------------------------------- classify

def classify_cat(slugged: str) -> Tuple[str, str]:
    if PUB_HINT.search(slugged):
        return "publication", slugged.replace("_", " ")
    if GENRE_HINT.search(slugged):
        return "genre", slugged.replace("_", " ")
    return "category", slugged.replace("_", " ")

# ----------------------------------------------------------------------------- load map

def load_existing(path: Path) -> set[Tuple[str, str]]:
    seen = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as fh:
        for raw, field, _ in csv.reader(fh):
            seen.add((raw, field))
    return seen

# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser("Stream corpus, enrich via Wikisource fingerprint")
    ap.add_argument("--xml", required=True, help="normalized_corpus.xml")
    ap.add_argument("--map", default="header_map.csv", help="existing header map")
    ap.add_argument("--out", default="wikisource_patch.csv", help="patch CSV output")
    args = ap.parse_args()

    xml_path = Path(args.xml)
    map_path = Path(args.map)
    out_path = Path(args.out)

    if not xml_path.exists():
        sys.exit(f"Input XML not found at '{xml_path}'")

    existing = load_existing(map_path)
    db = open_cache()
    sess = requests.Session()

    patch_rows = set()
    n_docs = sum(1 for l in open(xml_path, encoding="utf-8", errors="ignore") if "<doc " in l)

    with tqdm(total=n_docs, desc="Processing documents") as bar:
        for _, el in ET.iterparse(xml_path, events=("end",)):
            if el.tag == "doc":
                bar.update(1)

                title = el.get("title") or ""
                author = el.get("author")
                
                # --- Simplified text extraction for fingerprint ---
                text_for_fp = " ".join(el.itertext())
                fp = fingerprint(text_for_fp)

                pageid = find_page(title, author, fp, sess, db)
                if not pageid:
                    el.clear()
                    continue

                cats = pull_categories(pageid, sess, db)
                for cat in cats:
                    slugged = slug(cat)
                    if not slugged: continue
                    
                    field, val = classify_cat(slugged)
                    pair = (slugged, field)
                    if pair not in existing:
                        patch_rows.add((slugged, field, val))
            
            # --- Crucial memory leak fix ---
            el.clear()

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(sorted(patch_rows))
    print(f"✓ Collected {len(patch_rows):,} new rows → {out_path}")
    db.close()

if __name__ == "__main__":
    main()