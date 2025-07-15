# tools/export_topics.py
from pathlib import Path
import pandas as pd
import torch                                 # ← fix 1
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

RUN_DIR   = Path("outputs/20250715-143249")   # ← adjust timestamp
DOCS_FILE = Path("docs.txt")                  # same file used for training
TOPN      = 20                               # words per topic

# 1 · attach the same embedder you trained with
embedder = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    device="cuda" if torch.cuda.is_available() else "cpu"   # ← or just "cpu"
)

topic_model = BERTopic.load(RUN_DIR / "bertopic_model",
                            embedding_model=embedder)

# 2 · load docs, get assignments
docs = [line.split("\t", 1)[-1].strip() for line in DOCS_FILE.open(encoding="utf-8")]
topics, _ = topic_model.transform(docs)

pd.DataFrame({"doc_id": range(len(docs)), "topic": topics}) \
  .to_csv(RUN_DIR / "doc_topics.csv", index=False)

# 3 · top‑N words per topic
rows = []
for tid, words in topic_model.get_topics().items():
    if tid == -1:
        continue
    rows.append({"topic": tid,
                 **{f"word_{i+1}": w for i, (w, _) in enumerate(words[:TOPN])}})
pd.DataFrame(rows).to_csv(RUN_DIR / "topic_words.csv", index=False)

print("✓ wrote doc_topics.csv and topic_words.csv to", RUN_DIR)
