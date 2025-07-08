#!/usr/bin/env python3
"""
auto_title_seed.py  –  aggressively mark remaining tokens as TITLE
Run:
    python auto_title_seed.py annotated_corpus.xml        # write rows
    python auto_title_seed.py annotated_corpus.xml --dry  # counts only
"""
import argparse, csv, re, unicodedata
from pathlib import Path

MAP       = Path("header_map.csv")
DOC_RE    = re.compile(r"<doc\s+([^>]+)>", re.I)
ATTR_RE   = re.compile(r'(\w+)="([^"]+)"')

CENTURY   = re.compile(r"^\d{1,2}\.?\s*stoletje$", re.I)
INITIALS  = re.compile(r"^[A-ZČŠŽ]\.\s?\w+")
DIGIT     = re.compile(r"\d")
TRAILING  = re.compile(r"[ \t]*[.⋯…,:;]+$")

def slug(txt: str) -> str:
    norm = unicodedata.normalize("NFD", txt).encode("ascii","ignore").decode()
    return re.sub(r"[^\w]+", "_", norm.lower()).strip("_")

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

def should_skip(tok: str) -> bool:
    if "_" in tok:
        return True
    if DIGIT.search(tok) or CENTURY.match(tok):
        return True
    if INITIALS.match(tok):
        return True
    if tok.isupper() and len(tok) <= 5:
        return True
    if tok.islower():
        return True
    return False

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="+")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    # load known slugs
    known = set()
    if MAP.exists():
        with MAP.open(newline="", encoding="utf-8") as fh:
            for raw, _, _ in csv.reader(fh):
                known.add(raw)

    unknown = set()
    for file in args.corpus:
        with open(file, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                unknown.update(t for t in extract_tokens(line) if t not in known)

    to_title = []
    for tok in unknown:
        if should_skip(tok):
            continue
        clean = TRAILING.sub("", tok).strip()
        if clean:
            to_title.append((slug(clean), clean))

    print(f"Unknown tokens     : {len(unknown):,}")
    print(f"Marking as title   : {len(to_title):,}")

    if args.dry or not to_title:
        return

    added = 0
    with MAP.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for raw, val in sorted(to_title, key=lambda x: x[1].casefold()):
            if (raw, "title") not in known:
                w.writerow([raw, "title", val])
                added += 1
    print(f"✓ Appended {added} titles → {MAP}")

if __name__ == "__main__":
    main()
