# Slovene-Wikisource Corpus – Build & Analysis Toolkit

A reproducible pipeline for turning the messy Slovene Wikisource dump (\~50 M words, ≈ 23 k texts) into a **richly annotated research corpus** that feeds Classla, BERTopic, LDA and downstream linguistic/DH workflows.

---

## Repository layout

```
.
├─ header_map.csv               # master token → field → value map
├─ junk_tokens.txt              # slugs we always ignore
├─ normalized_corpus.xml        # clean XML corpus (rewritten <doc> headers)
│
├─ tools/
│   ├─ header_curator.py        # interactive one‑key tagger
│   ├─ mapper.py                # auto‑map new tokens
│   ├─ validate_header_map.py   # sanity checks
│   ├─ apply_header_map.py      # rewrite <doc …> headers (safe in‑place)
│   ├─ ws_crawler.py            # crawl authors/genres (category tree)
│   ├─ ws_fingerprint_enrich.py # overnight fingerprint‑based enrich
│   ├─ normalize_categories.py  # ASCII + spaces for category values
│   ├─ normalize_genres.py      # ASCII + spaces for genre values
│   ├─ run_classla_xml.py       # stream‑annotate with Classla
│   ├─ export_metadata_counts.py# coverage stats
│   └─ topic_model.py           # BERTopic/LDA (run in .topic_env)
└─ stats/                       # generated frequency tables
```

---

\## 1  Environment bootstrap

```bash
# clone repo
git clone https://github.com/yourname/wikivir-corpus.git
cd wikivir-corpus

# ----- metadata / crawling toolchain -----
python3.9 -m venv .meta
source .meta/bin/activate
pip install requests tqdm

# ----- linguistic annotation -------------
python3.9 -m venv .classla
source .classla/bin/activate
pip install classla tqdm
classla.download sl        # ~500 MB, once
deactivate

# ----- topic modelling -------------------
python3.9 -m venv .topic_env
source .topic_env/bin/activate
pip install bertopic sentence-transformers gensim scikit-learn
deactivate
```

---

\## 2  Header‑normalisation workflow

| phase                               | command(s)                                                                                                        | purpose                                                |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
|  **2.1** list tokens still unmapped | `python tools/list_remaining_tokens.py normalized_corpus.xml > unmapped.txt`                                      | produces *844* leftovers after first pass              |
|  **2.2** auto‑classify + curate     | `python tools/mapper.py unmapped.txt header_map.csv authors.txt cache.sqlite`<br>`python tools/header_curator.py` | auto maps obvious rows, then one‑key menu for the rest |
|  **2.3** validate                   | `python tools/validate_header_map.py header_map.csv normalized_corpus.xml`                                        | confirms “Corpus slugs still unmapped: 0”              |
|  **2.4** rewrite headers            | `python tools/apply_header_map.py --in-place normalized_corpus.xml`                                               | promotes all mapped tokens, drops junk                 |

### Field inventory

* **title**   (always present)
* **author**
* **year**     (4‑digit)
* **century**
* **genre**
* **publication**
* **category** (residual topical tags)
* **junk**     (optional noise bucket)

---

\## 3  Optional Wikisource enrichment

```bash
python tools/ws_fingerprint_enrich.py \
       --xml normalized_corpus.xml \
       --map header_map.csv \
       --out wikisource_patch.csv
cat wikisource_patch.csv >> header_map.csv
python tools/validate_header_map.py header_map.csv normalized_corpus.xml
python tools/apply_header_map.py --in-place normalized_corpus.xml
```

Fingerprint matching finds pages even if titles differ and classifies extra
categories into `publication | genre | category`.

---

\## 4  Classla annotation

```bash
source .classla/bin/activate
python tools/run_classla_xml.py normalized_corpus.xml \
       --out docs.txt --threads 32
deactivate
```

* processors: `tokenize,pos,lemma,ner`
* output: `docs.txt` (lemmatised + merged‑NER line per doc)

---

\## 5  Corpus statistics

```bash
source .meta/bin/activate
python tools/export_metadata_counts.py normalized_corpus.xml
open stats/      # docs_per_author.csv etc.
deactivate
```

---

\## 6  Topic modelling

```bash
source .topic_env/bin/activate
python tools/topic_model.py docs.txt \
       --method bertopic --min_topic_size 10 --out topics/
deactivate
```

Outputs:

* `topic_keywords.json` – top‑n words per topic
* `topic_docs.tsv` – doc → topic mapping
* `topic_over_time.csv` – if `year`/`century` present

Switch `--method lda` for gensim‑LDA fallback.

---

\## 7  Reproducibility notes

* All corpus scripts are **stream‑safe** (`xml.etree.iterparse`).
* Crawlers cache to `cache.sqlite` so runs are resumable and API‑friendly.
* Destructive steps write to temp files first (atomic replace) – safe against truncation.

---

\## 8  Next ideas

1. **Named‑entity grounding** via Wikidata IDs (already cached).
2. **Embedding explorer** (Doc2Vec / SentenceBERT) for nearest‑neighbour search.
3. **Streamlit dashboard** for interactive topic exploration.

Pull requests & issues welcome!
