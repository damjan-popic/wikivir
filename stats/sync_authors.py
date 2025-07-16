#!/usr/bin/env python3
"""
sync_authors.py – bring header_map.csv in line with authors.txt

Usage
-----
    python sync_authors.py authors.txt header_map.csv
"""
import csv, re, unicodedata, sys, shutil, pathlib

AUTHOR_FIELD = "author"

def slug(name: str) -> str:
    """ascii, lower, accents stripped, spaces->underscore."""
    n = unicodedata.normalize("NFD", name).encode("ascii","ignore").decode()
    return re.sub(r"[^\w]+", "_", n.lower()).strip("_")

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: sync_authors.py authors.txt header_map.csv")

    authors_file = pathlib.Path(sys.argv[1])
    csv_file     = pathlib.Path(sys.argv[2])

    # load canonical authors
    wanted = {}
    for line in authors_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            wanted[slug(name)] = name

    # read existing map
    rows, idx = [], {}
    with csv_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row: continue
            raw, field, val = row[:3]
            key = (raw, field)
            if key in idx:
                continue          # skip duplicate rows
            idx[key] = len(rows)
            rows.append([raw, field, val])

    added = updated = 0
    for raw, canon in wanted.items():
        key_author = (raw, AUTHOR_FIELD)
        # case 1: already correctly mapped
        if key_author in idx:
            i = idx[key_author]
            if rows[i][2] != canon:
                rows[i][2] = canon
                updated += 1
            continue

        # case 2: same raw but wrong field – rewrite
        wrong = [k for k in idx if k[0] == raw]
        if wrong:
            i = idx[wrong[0]]
            rows[i] = [raw, AUTHOR_FIELD, canon]
            # update index keys
            for k in wrong:
                del idx[k]
            idx[key_author] = i
            updated += 1
        else:
            # case 3: brand-new author
            rows.append([raw, AUTHOR_FIELD, canon])
            idx[key_author] = len(rows)-1
            added += 1

    # write backup then new csv
    shutil.copy(csv_file, csv_file.with_suffix(".bak"))
    with csv_file.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)

    print(f"✓ Sync complete – {added} added, {updated} updated. "
          f"Backup at {csv_file.with_suffix('.bak').name}")

if __name__ == "__main__":
    main()
