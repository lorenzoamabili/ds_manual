# 10 · NLP & Language AI

A horizontal specialisation used across every domain. The field moved from
hand-crafted features to embeddings to transformers to LLMs — but the older tools
are still the right choice for many jobs.

## The ladder of techniques (pick the lowest rung that solves the problem)

| Rung | Technique | Use when |
|------|-----------|----------|
| 1 | **Rules / regex / keyword** | Structured extraction, small closed problems; a strong, debuggable baseline |
| 2 | **Bag-of-words / TF-IDF + linear model** | Text classification with decent labelled data; fast, interpretable, hard to beat on narrow tasks |
| 3 | **Classical embeddings (word2vec, GloVe, fastText)** | Semantic similarity when you can't afford transformers |
| 4 | **Transformer encoders (BERT family, fine-tuned)** | Classification/NER/QA needing real language understanding; you have labels + GPUs |
| 5 | **Sentence embeddings (SBERT) + vector search** | Semantic search, clustering, dedup, retrieval |
| 6 | **LLMs (prompted or fine-tuned)** | Generation, summarisation, extraction, zero/few-shot on tasks without training data |

**Don't jump to rung 6 by default.** A TF-IDF + logistic regression model that you
can retrain in seconds and explain to a regulator often beats an LLM for
production classification on cost, latency, and reliability.

## Classic NLP tasks
Text classification, sentiment analysis, named-entity recognition (NER), topic
modelling (LDA, or BERTopic), summarisation, translation, and semantic search.
Preprocessing (tokenisation, lowercasing, lemmatisation, stop-words) still matters
for rungs 1–3; transformers largely handle it internally.

## Working with LLMs

- **Prompting first.** Zero-/few-shot prompting, clear instructions, and
  structured output (ask for JSON) solve a surprising fraction of tasks with no
  training.
- **RAG (Retrieval-Augmented Generation)** — the standard pattern for
  question-answering over *your* documents: embed the corpus, retrieve the
  relevant chunks at query time, and put them in the prompt. Grounds answers and
  reduces hallucination. The retrieval quality usually matters more than the model.
- **Fine-tuning** — reach for it when you need a consistent style/format, have many
  examples, or want a smaller/cheaper model to match a big one on a narrow task.
  Parameter-efficient methods (LoRA) make this cheap.
- **Evaluation is the hard part.** Generation has no single right answer. Use a
  mix: task-specific metrics, held-out human ratings, and "LLM-as-judge" (with
  care — it has biases). Build a small **golden set** of examples and never ship a
  prompt change without running it.

## Pitfalls specific to NLP
- **Hallucination** — LLMs state falsehoods fluently; ground with retrieval and
  cite sources.
- **Train/test contamination** — public benchmarks may be in the model's training
  data; your eval must use fresh, private examples.
- **Bias & toxicity** — models inherit their training data's biases; test for them.
- **Cost & latency** — a per-call LLM in a high-throughput loop is an expensive
  design; cache, batch, or distil.
- **Label leakage in classic pipelines** — fit TF-IDF/vectorisers on training folds
  only, like any other transform ([03](03-data-and-feature-engineering.md)).

---

## Python example — TF-IDF baseline + top-features inspection

```python
"""
NLP text classification: TF-IDF + Logistic Regression pipeline.
Demonstrates: vectoriser inside Pipeline (no leakage), feature inspection,
and why to try this before BERT.

Paired project: P9 (20 Newsgroups, 4 categories)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

CATS = ["sci.med", "sci.space", "rec.sport.baseball", "talk.politics.guns"]
train = fetch_20newsgroups(subset="train", categories=CATS,
                            remove=("headers","footers","quotes"))
test  = fetch_20newsgroups(subset="test",  categories=CATS,
                            remove=("headers","footers","quotes"))

# ── Pipeline: vectoriser fit ONLY on training data ────────────────────────────
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1,2), sublinear_tf=True)),
    ("clf",   LogisticRegression(C=1.0, max_iter=500, random_state=42)),
])
pipe.fit(train.data, train.target)
preds = pipe.predict(test.data)
print(classification_report(test.target, preds, target_names=train.target_names))

# ── Top discriminating features per class ────────────────────────────────────
vocab = pipe.named_steps["tfidf"].get_feature_names_out()
coef  = pipe.named_steps["clf"].coef_
TOP_N = 8

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, cls_idx in zip(axes, range(4)):
    top_idx   = np.argsort(coef[cls_idx])[-TOP_N:]
    top_words = vocab[top_idx]
    top_vals  = coef[cls_idx][top_idx]
    ax.barh(range(TOP_N), top_vals, color="#3498db")
    ax.set_yticks(range(TOP_N)); ax.set_yticklabels(top_words, fontsize=8)
    ax.set_title(train.target_names[cls_idx].split(".")[-1], fontsize=9)
plt.suptitle("Top TF-IDF features per class — interpretable NLP baseline")
plt.tight_layout()
plt.savefig("nlp_top_features.png", dpi=120)
plt.close()

# ── 5-fold CV to confirm ──────────────────────────────────────────────────────
from sklearn.datasets import fetch_20newsgroups
all_data = fetch_20newsgroups(subset="all", categories=CATS,
                               remove=("headers","footers","quotes"))
cv_scores = cross_val_score(pipe, all_data.data, all_data.target,
                             cv=5, scoring="f1_macro")
print(f"5-fold CV F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print("Beat this before reaching for BERT.")
```

---

## Cross-references

- [P9](../projects/p9_nlp_classification) — full NLP project (20 Newsgroups, 4-class)
- [37](37-cybersecurity.md) — log parsing and phishing NLP
- [33](33-healthtech.md) — clinical NLP
