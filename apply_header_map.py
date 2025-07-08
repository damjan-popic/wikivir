#!/usr/bin/env python3.9
"""
apply_header_map.py – normalize document headers using header_map.csv

Usage:
    python apply_header_map.py input.xml output.xml
    python apply_header_map.py --in-place input.xml

Reads header_map.csv in cwd for literal and regex rules,
filters out JUNK tokens listed in junk_tokens.txt,
and rewrites each <doc> header with promoted attributes.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# ------------ load JUNK set ------------------------------------------------
JUNK_FILE = Path('junk_tokens.txt')
if JUNK_FILE.exists():
    JUNK = set(JUNK_FILE.read_text(encoding='utf-8').splitlines())
else:
    JUNK = set()

# ------------ load mapping -------------------------------------------------
def load_maps(csv_path: Path):
    literals = {}
    regexes = []  # list of (pattern, field, value_template)
    for raw, field, val in csv.reader(csv_path.open(encoding='utf-8')):
        # skip duplicates
        if raw in literals:
            continue
        if raw.startswith('r"') or raw.startswith("r'"):
            # regex rule
            pat = raw[2:-1]
            regexes.append((re.compile(pat), field, val))
        else:
            literals[raw] = (field, val)
    return literals, regexes

# ------------ header processing --------------------------------------------
ATTR_RE = re.compile(r"(<doc)([^>]*?)>")


def process_header(line: str, literals: dict, regexes: list) -> str:
    """Promote mapped tokens to attributes, filter out JUNK, rebuild header line."""
    m = ATTR_RE.search(line)
    if not m:
        return line

    prefix, body = m.group(1), m.group(2)
    # extract existing attributes
    # find categories="..."
    cats_match = re.search(r'categories="([^"]*)"', body)
    categories = cats_match.group(1) if cats_match else ''
    # split and filter
    slugs = [tok.strip() for tok in categories.split(',') if tok.strip()]
    slugs = [tok for tok in slugs if tok not in JUNK]

    # build new attrs dict
    attrs = {}
    # carry over title, author, year, century if present
    for attr in ['title','author','year','century']:
        m2 = re.search(fr'{attr}="([^"]*)"', body)
        if m2:
            attrs[attr] = m2.group(1)

    # promote from slugs
    remaining = []
    for tok in slugs:
        if tok in literals:
            field, val = literals[tok]
            attrs[field] = val
        else:
            matched = False
            for pat, field, val in regexes:
                if pat.search(tok):
                    attrs[field] = pat.sub(val, tok)
                    matched = True
                    break
            if not matched:
                remaining.append(tok)

    # rebuild header
    parts = [prefix]
    for k,v in attrs.items():
        parts.append(f'{k}="{v}"')
    if remaining:
        parts.append('categories="' + ', '.join(remaining) + '"')
    parts.append('>')
    return ' '.join(parts) + '\n'

# ------------ file processing ----------------------------------------------

def process_file(src: Path, dst: Path, literals, regexes):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open('r', encoding='utf-8', errors='ignore') as inp, \
         dst.open('w', encoding='utf-8') as out:
        for line in inp:
            if line.lstrip().startswith('<doc '):
                line = process_header(line, literals, regexes)
            out.write(line)

# ------------ main ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Apply header_map.csv to XML corpus')
    ap.add_argument('--in-place', action='store_true', help='overwrite input file')
    ap.add_argument('input', help='path to input XML or corpus directory')
    ap.add_argument('output', nargs='?', help='path to output XML or directory')
    args = ap.parse_args()

    inp = Path(args.input)
    if args.in_place:
        out = inp.with_suffix('.tmp')
    else:
        if not args.output:
            sys.exit('Error: output path required when not using --in-place')
        out = Path(args.output)

    csv_path = Path('header_map.csv')
    if not csv_path.exists():
        sys.exit('header_map.csv not found in current directory.')

    literals, regexes = load_maps(csv_path)

    # single file only
    process_file(inp, out, literals, regexes)

    if args.in_place:
        out.replace(inp)
        print(f'✓ Replaced {inp} in-place')
    else:
        print(f'✓ Wrote cleaned file to {out}')

if __name__ == '__main__':
    main()
