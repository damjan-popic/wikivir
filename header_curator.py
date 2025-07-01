#!/usr/bin/env python3
"""header_curator.py – Interactive normaliser for messy <doc …> headers

Usage (basic):
    python header_curator.py corpus.xml                # interactive tagging

Key menu during the session (press **one** key, *no* Enter required):
    a  →  Author      (then **Enter** to accept the suggested canonical form)
    y  →  Year        (4-digit)
    c  →  Century     (1- or 2-digit; e.g. 19)
    n  →  Genre       (literary / journalistic genre)
    g  →  Category    (leave token in <categories>)
    t  →  Title       (rarely needed – *case preserved*)
    s  →  Skip        (treat as category, don’t ask again)
    q  →  Quit & save progress immediately

**Tip:** whenever the prompt shows something like::

    Canonical author [Jovan Vesel Koseski] (Enter = keep):

…just press **Enter** if the default looks good, or type the corrected value and hit Enter.

Every answer is flushed straight to *header_map.csv* (utf-8, *raw,field,value*) so you’re crash-safe.

Only std-lib modules are used (csv, re, argparse, readline, pathlib). Colour is added if
`colorama` is installed but the script works fine without it.
"""
from __future__ import annotations

import argparse
import csv
import re
import readline  # noqa: F401 – improves CLI editing on *nix/mac; harmless on Windows
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants & regexes
# ---------------------------------------------------------------------------
RAW_COL, FIELD_COL, VALUE_COL = 0, 1, 2
DEFAULT_MAP_FILE = "header_map.csv"
DOC_HEADER_RE = re.compile(r"<doc\s+([^>]*?)>", re.IGNORECASE)
ATTR_RE = re.compile(r"(\w+)=\"(.*?)\"")

# Pre-seed regex patterns that are obvious enough to skip manual tagging
# (pattern, field, value_template – {m} is replaced with the full match)
PRESEEDED_REGEXES: List[Tuple[str, str, str]] = [
    (r"^\d{4}$", "year", "{m}"),                                   # 4-digit year
    (r"^(1[0-9]|20|21|[1-9])$", "century", "{m}"),                # 1- or 2-digit century (1-21)
]

try:
    from colorama import Fore, Style, init as colorama_init  # type: ignore

    colorama_init()
    COLOUR = True
except ImportError:  # pragma: no cover – colour is optional
    class _Dummy:
        RESET = RED = GREEN = CYAN = YELLOW = ""

    Fore = Style = _Dummy()  # type: ignore
    COLOUR = False

# ---------------------------------------------------------------------------
# Map handling helpers
# ---------------------------------------------------------------------------

def load_map(path: Path) -> Tuple[Dict[str, Tuple[str, str]], List[Tuple[re.Pattern, str, str]]]:
    """Return (literal_map, regex_map).

    *literal_map[token] = (field, value)* for exact tokens
    *regex_map = [(compiled_pattern, field, value_template), …]*
    """
    literals: Dict[str, Tuple[str, str]] = {}
    regexes: List[Tuple[re.Pattern, str, str]] = []

    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for raw, field, value in reader:
                if raw.startswith("^"):           # treat as regex rule
                    regexes.append((re.compile(raw), field, value))
                else:
                    literals[raw] = (field, value)

    # Add pre-seeded regexes that aren’t already present
    for pat, field, value in PRESEEDED_REGEXES:
        if not any(r.pattern == pat for r, *_ in regexes):
            regexes.append((re.compile(pat), field, value))

    return literals, regexes


def append_map_row(path: Path, raw: str, field: str, value: str) -> None:
    """Append one classification row to *path* (always UTF-8, no header)."""
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow([raw, field, value])

# ---------------------------------------------------------------------------
# Token extraction from <doc …> headers
# ---------------------------------------------------------------------------

def extract_tokens(line: str) -> List[str]:
    """Pull candidate tokens from a single <doc …> header line."""
    m = DOC_HEADER_RE.search(line)
    if not m:
        return []

    attr_blob = m.group(1)
    tokens: List[str] = []

    for attr, raw_val in ATTR_RE.findall(attr_blob):
        val = raw_val.strip()
        if not val:
            continue
        if attr.lower() == "categories":          # comma-separated
            for tok in val.split(","):
                tok = tok.strip().lower()
                if tok:
                    tokens.append(tok)
        else:
            tokens.append(val.strip())
    return tokens

# ---------------------------------------------------------------------------
# Interactive classification UI
# ---------------------------------------------------------------------------
MENU = (
    f"{Fore.CYAN if COLOUR else ''}[a]{Style.RESET_ALL if COLOUR else ''} Author   "
    f"{Fore.CYAN if COLOUR else ''}[y]{Style.RESET_ALL if COLOUR else ''} Year   "
    f"{Fore.CYAN if COLOUR else ''}[c]{Style.RESET_ALL if COLOUR else ''} Century   "
    f"{Fore.CYAN if COLOUR else ''}[n]{Style.RESET_ALL if COLOUR else ''} Genre   "
    f"{Fore.CYAN if COLOUR else ''}[g]{Style.RESET_ALL if COLOUR else ''} Category   "
    f"{Fore.CYAN if COLOUR else ''}[t]{Style.RESET_ALL if COLOUR else ''} Title   "
    f"{Fore.CYAN if COLOUR else ''}[s]{Style.RESET_ALL if COLOUR else ''} Skip   "
    f"{Fore.CYAN if COLOUR else ''}[q]{Style.RESET_ALL if COLOUR else ''} Quit"
)

FIELD_BY_KEY = {
    "a": "author",
    "y": "year",
    "c": "century",
    "n": "genre",
    "g": "category",
    "t": "title",
}


def classify_token(token: str, total_left: int) -> Tuple[str, str] | None:
    """Ask the user how to tag *token*; return (field, value) or **None** to quit."""
    print("\n" + "─" * 64)
    print(f"RAW → {Fore.GREEN if COLOUR else ''}{token}{Style.RESET_ALL if COLOUR else ''}")
    print(MENU)
    while True:
        ch = input("Choice: ").strip().lower()[:1]
        if ch == "q":
            return None
        elif ch == "s":
            return "category", token
        elif ch in FIELD_BY_KEY:
            field = FIELD_BY_KEY[ch]
            # ----------------------------------------------------------------
            # Determine default canonical form per field
            # ----------------------------------------------------------------
            if field == "author":
                default = token.replace("_", " ").title()
                prompt = f"Canonical author [{default}] (Enter = keep): "
                val = input(prompt).strip() or default
            elif field == "title":
                default = token.replace("_", " ")  # keep original case
                prompt = f"Canonical title [{default}] (Enter = keep): "
                val = input(prompt).strip() or default
            elif field == "year":
                match = re.search(r"\d{4}", token)
                default = match.group(0) if match else token
                prompt = f"Year (4-digits) [{default}] (Enter = keep): "
                val = input(prompt).strip() or default
            elif field == "century":
                default = token
                prompt = f"Century [{default}] (Enter = keep): "
                val = input(prompt).strip() or default
            else:  # genre / category – no canonicalisation needed
                val = token
            return field, val
        else:
            print("Invalid key; please choose a valid option (see menu above).")

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Interactive header-token classifier")
    p.add_argument("files", metavar="FILE", nargs="+", help="Corpus XML/JSONL/… files to scan")
    p.add_argument("--map", dest="map_path", default=DEFAULT_MAP_FILE, help="Path to header_map.csv")
    args = p.parse_args(argv)

    map_path = Path(args.map_path)
    literals, regexes = load_map(map_path)

    # -----------------------------------------------------------------------
    # Collect unique tokens from corpus files
    # -----------------------------------------------------------------------
    unique: List[str] = []
    seen: set[str] = set()
    for file in args.files:
        with open(file, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for tok in extract_tokens(line):
                    if tok not in seen:
                        seen.add(tok)
                        unique.append(tok)

    # Filter out anything already covered by literals or regexes
    def is_known(tok: str) -> bool:
        if tok in literals:
            return True
        return any(pat.fullmatch(tok) for pat, *_ in regexes)

    unknown = [tok for tok in unique if not is_known(tok)]

    print(f"Loaded {len(unique)} unique tokens from corpus – {len(unique) - len(unknown)} already classified, {len(unknown)} left.")

    # -----------------------------------------------------------------------
    # Interactive loop
    # -----------------------------------------------------------------------
    total_left = len(unknown)
    for token in unknown:
        res = classify_token(token, total_left)
        if res is None:
            print("\nSession ended by user. Progress saved. Bye!")
            break
        field, value = res
        append_map_row(map_path, token, field, value)
        literals[token] = (field, value)
        total_left -= 1
        print(f"✅ Saved – {total_left} tokens left to classify.")

    if total_left == 0:
        print("\nAll tokens classified – nice work!")


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – progress saved where possible.")
        sys.exit(1)
