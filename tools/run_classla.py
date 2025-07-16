#!/usr/bin/env python3.9
"""
run_classla.py – Reliably stream-annotate a Slovene XML corpus.

This version is namespace-aware and robustly parses standard XML,
ensuring each document is processed independently to generate a clean
output file for topic modeling.
"""
from __future__ import annotations
import argparse
import xml.etree.ElementTree as ET
import sys
import logging
import time
from pathlib import Path
import classla
from tqdm import tqdm

def count_docs_reliably(path: Path) -> int:
    """
    FIX: Accurately count <doc> elements, ignoring XML namespaces.
    This is much more reliable than line-based counting.
    """
    count = 0
    for event, elem in ET.iterparse(path, events=("end",)):
        # Check the local name of the tag, ignoring any {namespace} part
        if elem.tag.rsplit('}', 1)[-1] == "doc":
            count += 1
        # Clear element to keep memory usage low during the count
        elem.clear()
    return count

def merge_ner(doc: classla.Document) -> str:
    """Return lemmas, merging multi-word named entities with underscores."""
    out, span, label = [], [], None
    keep = {"NOUN", "PROPN", "ADJ", "VERB"}
    for sent in doc.sentences:
        for tok in sent.tokens:
            ner = tok.ner
            word = tok.words[0]
            if ner.startswith("B-"):
                if span:
                    out.append("_".join(span))
                label = ner[2:]
                span = [word.lemma]
            elif ner.startswith("I-") and label == ner[2:]:
                span.append(word.lemma)
            else:
                if span:
                    out.append("_".join(span))
                    span, label = [], None
                if word.upos in keep and len(word.lemma) > 1:
                    out.append(word.lemma)
    if span:
        out.append("_".join(span))
    return " ".join(out)

def extract_text(el: ET.Element) -> str:
    """
    Extract all text, joining fragments with newlines to preserve
    internal document structure for better sentence splitting.
    """
    return "\n".join(el.itertext()).strip()

def main() -> None:
    ap = argparse.ArgumentParser(description="Reliable Classla XML annotator.")
    ap.add_argument("corpus", help="Path to input XML corpus")
    ap.add_argument("--out", default="docs.txt", help="Path to output file")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    logging.info("Initializing Classla...")
    nlp = classla.Pipeline(lang="sl", processors="tokenize,pos,lemma,ner")

    src = Path(args.corpus)
    if not src.exists():
        sys.exit(f"Error: Input file not found at '{src}'")

    logging.info("Accurately counting documents (namespace-aware)...")
    est_docs = count_docs_reliably(src)
    logging.info(f"Found {est_docs:,} documents to process.")

    start_time = time.time()
    
    with open(args.out, "w", encoding="utf-8") as fout, \
         tqdm(total=est_docs, unit="doc", desc="Annotating") as bar:
        
        for event, elem in ET.iterparse(src, events=("end",)):
            # FIX: Use a namespace-aware check for the tag name.
            if elem.tag.rsplit('}', 1)[-1] == "doc":
                txt = extract_text(elem)
                if txt:
                    try:
                        doc = nlp(txt)
                        fout.write(merge_ner(doc) + "\n")
                    except Exception as e:
                        logging.error(f"Failed to process a document: {e}")
                bar.update(1)
            
            # This is the crucial fix that prevents memory leaks.
            elem.clear()

    total_time = time.time() - start_time
    logging.info(f"✓ Completed in {total_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()