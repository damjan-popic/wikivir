#!/usr/bin/env python3.9
"""
ws_crawler.py – build cache.sqlite with Slovene Wikisource authors & genres

Run
----
    python ws_crawler.py          # create / update if not present
    python ws_crawler.py --refresh   # force full re-crawl

Creates cache.sqlite with tables:
    authors(slug TEXT PK, name TEXT)
    genres (slug TEXT PK, name TEXT)
    meta   (key TEXT PK, value TEXT)   -- crawl timestamp
"""
from __future__ import annotations
import argparse, sqlite3, time, sys, requests, itertools
from pathlib import Path
from tqdm import tqdm

API   = "https://sl.wikisource.org/w/api.php"
HEAD  = {"User-Agent": "wikivir-crawler/0.4 (damjan@example.com)"}
DB    = Path("cache.sqlite")
SLEEP = 1.0                # seconds between requests

def slug(title: str) -> str:
    return title.lower().replace(" ", "_")

# ------------------------------------------------------------------ DB setup
def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS authors(
        slug TEXT PRIMARY KEY,
        name TEXT
    );
    CREATE TABLE IF NOT EXISTS genres(
        slug TEXT PRIMARY KEY,
        name TEXT
    );
    CREATE TABLE IF NOT EXISTS meta(
        key TEXT PRIMARY KEY,
        value TEXT
    );
    """)
    conn.commit()

def store(conn: sqlite3.Connection, table: str, names: set[str]):
    cur = conn.cursor()
    cur.executemany(
        f"INSERT OR IGNORE INTO {table}(slug, name) VALUES(?, ?)",
        [(slug(n), n) for n in names]
    )
    conn.commit()

# ------------------------------------------------------------------ API util
def api_get(sess: requests.Session, params: dict, tries=3):
    """GET with retry/back-off."""
    for attempt in range(tries):
        try:
            r = sess.get(API, params=params, headers=HEAD, timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            time.sleep(SLEEP * 2 ** attempt)

def category_members(sess: requests.Session, cat: str):
    """Yield page titles (no namespace) of a Wikisource category."""
    cmcontinue = None
    while True:
        p = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{cat}",
            "cmlimit": 500,
            "cmnamespace": 0,
            "format": "json",
        }
        if cmcontinue:
            p["cmcontinue"] = cmcontinue
        data = api_get(sess, p)
        for m in data["query"]["categorymembers"]:
            yield m["title"]
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(SLEEP)

# ------------------------------------------------------------------ crawl
def crawl(refresh: bool):
    conn = sqlite3.connect(DB)
    init_db(conn)
    cur = conn.cursor()

    ts = cur.execute("SELECT value FROM meta WHERE key='timestamp'").fetchone()
    if ts and not refresh:
        print(f"Cache already present (last crawl: {ts[0]}). "
              f"Use --refresh to rebuild.")
        conn.close()
        return

    with requests.Session() as S:
        # ---------------- AUTHORS ----------------
        author_roots = ["Avtorji", "Avtorice", "Slovenski_avtorji"]
        authors = set()
        print("Fetching authors …")
        for cat in tqdm(author_roots, desc="Author roots"):
            authors.update(category_members(S, cat))
        store(conn, "authors", authors)
        print("Authors stored:", len(authors))

        # ---------------- GENRES -----------------
        genre_roots = ["Poezija", "Proza", "Romani", "Povesti", "Drame"]
        genres = set()
        print("Fetching genres …")
        for root in tqdm(genre_roots, desc="Genre roots"):
            # depth 0 and 1
            level0 = set(category_members(S, root))
            genres.update(g for g in level0 if "_" in g)
            for sub in level0:
                genres.update(g for g in category_members(S, sub) if "_" in g)
        store(conn, "genres", genres)
        print("Genres stored:", len(genres))

    cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('timestamp',?)",
                (time.strftime("%Y-%m-%d %H:%M:%S"),))
    conn.commit()
    conn.close()
    print("✓ cache.sqlite ready!")

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="Force full re-crawl even if cache exists")
    args = ap.parse_args()
    try:
        crawl(refresh=args.refresh)
    except Exception as e:
        sys.exit(f"✗ Crawler failed: {e}")
