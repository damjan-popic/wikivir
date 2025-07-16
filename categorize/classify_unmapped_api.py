#!/usr/bin/env python3.9
"""
classify_unmapped_api.py  –  turn the unmapped tokens into header_map rows

Usage
-----
    python classify_unmapped_api.py unmapped.txt header_map.csv authors.txt

Outputs
-------
    patch_header_map.csv   ← append to header_map.csv
    review_needed.txt      ← tokens still ambiguous
"""
from __future__ import annotations
import csv, re, unicodedata, sys, pathlib, requests
from functools import lru_cache
from typing import List, Tuple
from tqdm import tqdm

# ---------- util ------------------------------------------------------------
SLUG_RGX = re.compile(r"[^\w]+")
def slug(s:str)->str:
    return SLUG_RGX.sub("_", unicodedata.normalize("NFD",s)
                        .encode("ascii","ignore").decode().lower()).strip("_")

CENTURY  = re.compile(r"^(\d{1,2})(?:_|\.?)(stoletje|stolje)", re.I)
YEAR4    = re.compile(r"\b(\d{4})\b")
SHELF    = re.compile(r"^dela(_[a-z])?$|^dela[a-z]?$", re.I)

def looks_title(tok:str)->bool:
    words = tok.replace("_"," ").split()
    return len(words)>=3 and words[0][0].isupper() and any(w[0].islower() for w in words[1:])

# ---------- local author set -----------------------------------------------
def load_known_authors(header_csv: pathlib.Path, authors_txt: pathlib.Path)->dict[str,str]:
    known={}
    # from header_map.csv
    with header_csv.open(newline="",encoding="utf-8") as fh:
        for raw,field,val in csv.reader(fh):
            if field=="author":
                known[raw]=val
    # from authors.txt
    for name in authors_txt.read_text(encoding="utf-8").splitlines():
        name=name.strip()
        if name:
            known[slug(name)]=name
    return known

# ---------- Wikimedia API ---------------------------------------------------
API = "https://sl.wikisource.org/w/api.php"
HEAD = {"User-Agent": "wikivir-meta/0.2 (damjan@example.com)"}

@lru_cache(maxsize=4096)
def wikimedia_says_author(slugged:str)->bool:
    """True if page is in Avtorji or has {{Avtor}}."""
    title = slugged.replace("_"," ")
    params={"action":"query","titles":title,
            "prop":"categories|templates",
            "cllimit":"max","tllimit":"max","format":"json"}
    try:
        data=requests.get(API,params=params,headers=HEAD,timeout=6).json()
        page=next(iter(data["query"]["pages"].values()))
        cats={c["title"].split(":",1)[-1].lower() for c in page.get("categories",[])}
        if any("avtor" in c for c in cats):
            return True
        tpls={t["title"].split(":",1)[-1].lower() for t in page.get("templates",[])}
        return "avtor" in tpls
    except Exception:
        return False

# ---------- classification --------------------------------------------------
def classify_piece(tok:str, known_auth:dict[str,str]) -> Tuple[str,str,str]|None:
    """Return (raw, field, value) or None to skip."""
    if SHELF.match(tok):
        return None
    if m:=YEAR4.search(tok):
        return tok,"year",m.group(1)
    if m:=CENTURY.match(tok):
        return tok,"century",m.group(1)
    if tok in known_auth:
        return tok,"author",known_auth[tok]
    if wikimedia_says_author(tok):
        return tok,"author",tok.replace("_"," ").title()
    if looks_title(tok):
        return tok,"title",tok.replace("_"," ").capitalize()
    return tok,"category",tok.replace("_"," ")

def classify_token(raw:str, known_auth)->List[Tuple[str,str,str]]:
    pieces=[p for p in raw.split("_kategorija") if p]
    rows=[]
    for part in pieces:
        row=classify_piece(part, known_auth)
        if row:
            rows.append(row)
    return rows

# ---------- main ------------------------------------------------------------
def main():
    if len(sys.argv)!=4:
        sys.exit("Usage: classify_unmapped_api.py unmapped.txt header_map.csv authors.txt")

    unmapped  = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    headercsv = pathlib.Path(sys.argv[2])
    authorstxt= pathlib.Path(sys.argv[3])
    known_auth= load_known_authors(headercsv, authorstxt)

    patch=set(); review=[]
    for raw in tqdm(unmapped, desc="Classifying"):
        rows=classify_token(raw, known_auth)
        if rows:
            patch.update(rows)
        else:
            review.append(raw)

    # write outputs
    with open("patch_header_map.csv","w",newline="",encoding="utf-8") as fh:
        csv.writer(fh).writerows(sorted(patch))
    print(f"✓ patch_header_map.csv written – {len(patch):,} rows")

    if review:
        pathlib.Path("review_needed.txt").write_text("\n".join(sorted(set(review))),
                                                     encoding="utf-8")
        print(f"⚠  {len(review)} tokens need manual tagging → review_needed.txt")

if __name__=="__main__":
    main()
