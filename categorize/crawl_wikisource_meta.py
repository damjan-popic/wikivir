#!/usr/bin/env python3.9
"""
crawl_wikisource_meta.py  –  overnight crawler that enriches header_map.csv
with fresh metadata from Slovene Wikisource.

Strategy
--------
* Stream the already‑clean **normalized_corpus.xml** (constant memory).
* For every <doc title="…" author="…"> attempt to find the matching
  Wikisource page **once** (cached in cache.sqlite).
* If found, pull:
    • categories  → potential genre / publication / category rows
    • templates   → detect {{Avtor}}, {{Delo}}
    • pageprops   → wikidata item (future use)
* Emit *new* rows only (skip token-field pairs already in header_map.csv).
* Rate‑limit to 1 request/sec.  The whole 23 k docs ≈ 7 h but resume‑able.

Usage
-----
    python crawl_wikisource_meta.py \
        --xml normalized_corpus.xml \
        --map header_map.csv \
        --out wikisource_patch.csv

You can interrupt anytime; progress is stored in cache.sqlite.
The next morning just

    cat wikisource_patch.csv >> header_map.csv
    python validate_header_map.py header_map.csv normalized_corpus.xml

Dependencies: requests, tqdm, sqlite3 (std‑lib).
"""
from __future__ import annotations
import argparse, csv, re, sqlite3, sys, time, unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
from tqdm import tqdm

API = "https://sl.wikisource.org/w/api.php"
HEAD = {"User-Agent": "wikivir-ws-meta/0.1 (damjan@example.com)"}
RATE = 1.0  # sec between API calls

SLUG_RGX = re.compile(r"[^\w]+")

def slug(s: str) -> str:
    return SLUG_RGX.sub("_", unicodedata.normalize("NFD", s)
                        .encode("ascii", "ignore").decode().lower()).strip("_")

def open_cache():
    db = sqlite3.connect("cache.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS pages(title TEXT PRIMARY KEY, pageid INT, ts REAL)")
    return db

# ---------------------------------------------------------------------------

def find_page(title: str, author: str | None, sess: requests.Session, cache: sqlite3.Connection):
    row = cache.execute("SELECT pageid FROM pages WHERE title=?", (title,)).fetchone()
    if row:
        return row[0]  # may be None if previous lookup failed

    # Search query – include author for disambiguation when available
    query = f'intitle:"{title}"'
    if author:
        query += f' incategory:"{author.split()[0]}"'
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
    }
    time.sleep(RATE)
    try:
        data = sess.get(API, params=params, headers=HEAD, timeout=20).json()
        hits = data["query"]["search"]
        pageid = hits[0]["pageid"] if hits else None
    except Exception:
        pageid = None
    cache.execute("INSERT OR REPLACE INTO pages(title,pageid,ts) VALUES(?,?,?)",
                  (title, pageid, time.time()))
    cache.commit()
    return pageid


def page_meta(pageid: int, sess: requests.Session):
    params = {
        "action": "query",
        "pageids": pageid,
        "prop": "categories|templates|pageprops",
        "cllimit": "max",
        "tllimit": "max",
        "format": "json",
    }
    time.sleep(RATE)
    data = sess.get(API, params=params, headers=HEAD, timeout=20).json()
    return next(iter(data["query"]["pages"].values()))

# ---------------------------------------------------------------------------

def load_existing_map(path: Path):
    literals = set()
    with path.open(encoding="utf-8") as fh:
        for raw, field, _ in csv.reader(fh):
            literals.add((raw, field))
    return literals

# naive rule sets
PUB_HINT = re.compile(r"vestnik|glas|list|zalozba|casopis")
GENRE_HINT = re.compile(r"romani|poezija|poezije|pesmi|drame|povesti|novela|humoreska|potopis|ess?ej")


def classify_cat(cat_slug: str):
    if PUB_HINT.search(cat_slug):
        return "publication", cat_slug.replace("_", " ")
    if GENRE_HINT.search(cat_slug):
        return "genre", cat_slug.replace("_", " ")
    return "category", cat_slug.replace("_", " ")

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Crawl Wikisource metadata and emit patch CSV")
    ap.add_argument("--xml", required=True, help="normalized_corpus.xml")
    ap.add_argument("--map", default="header_map.csv", help="existing header_map.csv")
    ap.add_argument("--out", default="wikisource_patch.csv", help="patch to write")
    args = ap.parse_args()

    xml_path = Path(args.xml); map_path = Path(args.map); out_path = Path(args.out)
    existing = load_existing_map(map_path)

    cache = open_cache()
    sess = requests.Session()

    patch_rows = set()

    # estimate docs for progress bar
    n_docs = sum(1 for l in open(xml_path, encoding="utf-8", errors="ignore") if "<doc " in l)

    with tqdm(total=n_docs, desc="Docs processed") as bar:
        for ev, el in ET.iterparse(xml_path, events=("end",)):
            if el.tag != "doc":
                continue
            title = el.get("title") or ""
            author = el.get("author")
            pageid = find_page(title, author, sess, cache)
            if pageid:
                meta = page_meta(pageid, sess)
                cats = [c["title"].split(":",1)[-1] for c in meta.get("categories", [])]
                for c in cats:
                    slugged = slug(c)
                    field, val = classify_cat(slugged)
                    if (slugged, field) not in existing:
                        patch_rows.add((slugged, field, val))
            bar.update(1)
            el.clear()

    # write patch CSV
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(sorted(patch_rows))
    print(f"✓ wrote {len(patch_rows):,} new rows → {out_path}")
    if patch_rows:
        print("Append to header_map.csv and re‑validate when ready.")

if __name__ == "__main__":
    main()