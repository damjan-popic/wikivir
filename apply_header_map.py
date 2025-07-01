#!/usr/bin/env python3
"""apply_header_map.py – Batch‑normalise <doc …> headers using header_map.csv

Typical usage:
    # Write cleaned files to ./clean relative to originals
    python apply_header_map.py corpus/**/*.xml

    # Overwrite originals *after* a backup – be careful!
    python apply_header_map.py --in‑place corpus/**/*.xml

The script:
    • Loads *header_map.csv* (exact + regex rules) – same format as header_curator.
    • Scans each <doc …> header, promoting tokens into proper attributes
      (author, year, century, genre) and removing them from <categories>.
    • Fills in *century* from *year* if still missing.
    • Rewrites the file either in‑place or into a parallel directory tree
      under ./clean/ (default).
    • Logs any still‑unmapped tokens to *unmapped.log* for later review.

Only std‑lib modules; no external deps.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Settings & regex helpers
# ---------------------------------------------------------------------------
DEFAULT_MAP_FILE = "header_map.csv"
OUT_DIR = "clean"
DOC_HEADER_RE = re.compile(r"<doc\s+([^>]*?)>")
ATTR_RE = re.compile(r"(\w+)=\"(.*?)\"")

# ---------------------------------------------------------------------------
# Mapping loader (shared with header_curator)
# ---------------------------------------------------------------------------

def load_map(path: Path) -> Tuple[Dict[str, Tuple[str, str]], List[Tuple[re.Pattern, str, str]]]:
    literals: Dict[str, Tuple[str, str]] = {}
    regexes: List[Tuple[re.Pattern, str, str]] = []
    if not path.exists():
        print(f"⚠️  Map file {path} not found – continuing with empty map.")
        return literals, regexes

    with path.open(newline="", encoding="utf-8") as fh:
        for raw, field, value in csv.reader(fh):
            if raw.startswith("^"):
                regexes.append((re.compile(raw), field, value))
            else:
                literals[raw] = (field, value)
    return literals, regexes

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def classify_token(token: str, literals: Dict[str, Tuple[str, str]], regexes: List[Tuple[re.Pattern, str, str]]):
    """Return (field, value) or ("category", token) if unmapped."""
    # Exact first
    if token in literals:
        return literals[token]
    # Regex second
    for pat, field, val_tpl in regexes:
        m = pat.fullmatch(token)
        if m:
            val = val_tpl.format(m=m.group(0), **m.groupdict())
            return field, val
    # Default – leave as category
    return "category", token


def compute_century(year: str) -> str:
    try:
        y = int(year)
        return str((y - 1) // 100 + 1)
    except ValueError:
        return ""


# Attribute output order for prettiness
ATTR_ORDER = [
    "title",
    "author",
    "year",
    "century",
    "genre",
    "categories",
]

# ---------------------------------------------------------------------------
# Processing of a single header line
# ---------------------------------------------------------------------------

def process_header(line: str, literals, regexes, unmapped: set[str]):
    m = DOC_HEADER_RE.search(line)
    if not m:
        return line  # untouched

    attr_blob = m.group(1)
    attrs = {k.lower(): v for k, v in ATTR_RE.findall(attr_blob)}

    # Tokenise categories (lower‑case)
    cat_tokens = []
    if "categories" in attrs and attrs["categories"]:
        cat_tokens = [t.strip().lower() for t in attrs["categories"].split(",") if t.strip()]

    # Classify each token in categories
    remaining_cats: List[str] = []
    for tok in cat_tokens:
        field, value = classify_token(tok, literals, regexes)
        if field == "category":
            remaining_cats.append(tok)  # keep
        else:
            # Promote
            if field not in attrs or not attrs[field]:
                attrs[field] = value
            # If attr exists but short (e.g. "Kosovel"), prefer canonical if matches
            elif field == "author" and len(attrs[field]) < len(value):
                attrs[field] = value

    # Also try to normalise existing author / title tokens directly
    for key in ["author", "genre", "century", "year"]:
        if key in attrs and attrs[key]:
            tok_lc = attrs[key].lower().replace(" ", "_")
            field, value = classify_token(tok_lc, literals, regexes)
            if field == key:
                attrs[key] = value

    # Deduce century from year if needed
    if "year" in attrs and attrs.get("year") and not attrs.get("century"):
        c = compute_century(attrs["year"])
        if c:
            attrs["century"] = c

    # Reconstruct categories attribute
    if remaining_cats:
        attrs["categories"] = ", ".join(remaining_cats)
    elif "categories" in attrs:
        del attrs["categories"]

    # Track unmapped tokens for report
    for tok in remaining_cats:
        unmapped.add(tok)

    # Rebuild header line
    ordered = []
    for key in ATTR_ORDER:
        if key in attrs:
            ordered.append(f'{key}="{attrs[key]}"')
    # plus any extras we didn't know about
    for key, val in attrs.items():
        if key not in ATTR_ORDER:
            ordered.append(f'{key}="{val}"')
    new_header = "<doc " + " ".join(ordered) + ">"
    return line.replace(m.group(0), new_header, 1)

# ---------------------------------------------------------------------------
# File‑level driver
# ---------------------------------------------------------------------------

def process_file(path: Path, out_path: Path, literals, regexes, unmapped):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r", encoding="utf-8", errors="ignore") as inp, out_path.open("w", encoding="utf-8") as outp:
        for line in inp:
            if "<doc" in line:
                line = process_header(line, literals, regexes, unmapped)
            outp.write(line)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Batch‑normalise corpus headers using header_map.csv")
    ap.add_argument("files", nargs="+", help="Input corpus files (supports wildcards if shell does)")
    ap.add_argument("--map", default=DEFAULT_MAP_FILE, help="Path to header_map.csv")
    ap.add_argument("--in‑place", action="store_true", help="Overwrite files instead of writing to ./clean/")
    args = ap.parse_args(argv)

    literals, regexes = load_map(Path(args.map))

    unmapped: set[str] = set()
    total = len(args.files)
    for idx, file in enumerate(args.files, 1):
        in_path = Path(file)
        if args.in_place:
            out_path = in_path
        else:
            out_path = Path(OUT_DIR) / in_path
        process_file(in_path, out_path, literals, regexes, unmapped)
        print(f"[{idx}/{total}] {in_path} → {out_path}")

    # Write unmapped report
    if unmapped:
        with open("unmapped.log", "w", encoding="utf-8") as fh:
            for tok in sorted(unmapped):
                fh.write(tok + "\n")
        print(f"⚠️  {len(unmapped)} tokens remain unmapped – see unmapped.log")
    else:
        print("✅ All tokens mapped – corpus clean!")


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted – exiting early.")
        sys.exit(1)
