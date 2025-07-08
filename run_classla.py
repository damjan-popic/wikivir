#!/usr/bin/env python3.9
"""
run_classla_xml.py – Stream-annotate Slovene corpus with Classla 2.2

• processors: tokenize,pos,lemma,ner (no depparse)
• streams the XML (constant memory)
• GPU speeds up NER (falls back to CPU)
• batching leverages GPU throughput if supported
• error-resilient per-document
• logs to stderr, with performance metrics
"""
from __future__ import annotations
import argparse
import xml.etree.ElementTree as ET
import sys
import re
import logging
import time
from pathlib import Path
import classla
from tqdm import tqdm

# Default batch size for GPU throughput
DEFAULT_BATCH_SIZE = 128

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
    """Extract all text from an XML element, joining fragments with spaces."""
    return " ".join(el.itertext()).strip()

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream-annotate a Slovene XML corpus with Classla."
    )
    ap.add_argument("corpus", help="Path to input corpus (normalized_corpus.xml)")
    ap.add_argument("--out", default="docs.txt", help="Path to output file")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help=f"Batch size for GPU (default {DEFAULT_BATCH_SIZE})")
    args = ap.parse_args()

    # --- Logging Setup ---
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    logging.info("Initializing Classla (GPU auto-detect)…")
    nlp = classla.Pipeline(
        lang="sl",
        processors="tokenize,pos,lemma,ner",
        use_gpu=True,
        tokenize_no_ssplit=True,
        verbose=False,
    )

    src = Path(args.corpus)
    if not src.exists():
        sys.exit(f"Error: Input file not found at '{src}'")

    # --- Pre-computation ---
    logging.info("Estimating document count...")
    regex = re.compile(r"<doc\b")
    with open(src, encoding="utf-8", errors="ignore") as f:
        est_docs = sum(1 for line in f if regex.search(line)) or 1
    logging.info(f"Estimated {est_docs:,} documents to process.")

    batch_size = args.batch_size
    logging.info(f"Using batch size: {batch_size}")

    start_time = time.time()
    
    # --- Main Processing Loop ---
    with open(args.out, "w", encoding="utf-8") as fout, \
         tqdm(total=est_docs, unit="doc", desc="Annotating", file=sys.stdout) as bar:
        
        batch_texts: list[str] = []
        
        # iterparse the XML file to stream-process it
        for event, elem in ET.iterparse(src, events=("end",)):
            if elem.tag == "doc":
                txt = extract_text(elem)
                if txt:
                    batch_texts.append(txt)

                # Process a batch when it's full
                if len(batch_texts) >= batch_size:
                    try:
                        docs = nlp(batch_texts)
                        for doc_obj in docs:
                            fout.write(merge_ner(doc_obj) + "\n")
                    except Exception as e:
                        logging.error(f"Failed to process a batch: {e}")
                    finally:
                        bar.update(len(batch_texts))
                        batch_texts.clear()
            
            # This is the crucial fix: clear EVERY element after processing
            # to prevent the memory leak.
            elem.clear()

        # Process the final batch if any documents are left over
        if batch_texts:
            try:
                docs = nlp(batch_texts)
                for doc_obj in docs:
                    fout.write(merge_ner(doc_obj) + "\n")
            except Exception as e:
                logging.error(f"Failed to process the final batch: {e}")
            finally:
                bar.update(len(batch_texts))

    total_docs = bar.n
    total_time = time.time() - start_time
    avg_time = total_time / max(total_docs, 1)
    logging.info(f"Total processed: {total_docs:,}/{est_docs:,} docs in {total_time:.2f}s "
                 f"(avg {avg_time:.3f}s per doc)")
    logging.info("✓ Completed annotation run.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user.")
    except Exception as e:
        err_msg = str(e)
        if "urlopen error" in err_msg or "CERTIFICATE_VERIFY_FAILED" in err_msg:
            print(
                "\nError: Could not download Classla models. Check your internet connection and SSL certificates.",
                file=sys.stderr,
            )
        else:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)