#!/usr/bin/env python3
"""
Grab every category title from Slovene Wikisource and write them to
categories_wikisource.txt  (one per line, Unicode NFC).

Requirements:
    pip install requests tqdm

Usage:
    python grab_wikisource_cats.py
"""

import requests, sys, time, unicodedata
from pathlib import Path
from tqdm import tqdm

API = "https://sl.wikisource.org/w/api.php"
OUT = Path("categories_wikisource.txt")
SLEEP = 0.2          # courteous pause between calls

def fetch_all_categories():
    params = {
        "action": "query",
        "list": "allcategories",
        "aclimit": "500",
        "format": "json",
    }
    cats = []
    print("Fetching category list from Slovene Wikisource …")
    pbar = tqdm(unit="cat")
    while True:
        r = requests.get(API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        batch = [c["*"] for c in data["query"]["allcategories"]]
        cats.extend(batch)
        pbar.update(len(batch))
        if "continue" in data:
            params.update(data["continue"])
            time.sleep(SLEEP)
        else:
            break
    pbar.close()
    return cats

def main():
    cats = fetch_all_categories()
    # NFC normalisation → consistent accents when you slug later
    cats = [unicodedata.normalize("NFC", c) for c in cats]
    OUT.write_text("\n".join(sorted(cats, key=str.casefold)), encoding="utf-8")
    print(f"✓ Saved {len(cats):,} categories to {OUT}")

if __name__ == "__main__":
    sys.exit(main())
