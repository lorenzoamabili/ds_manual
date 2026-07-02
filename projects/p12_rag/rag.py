"""
P12 · Retrieval-Augmented Generation (RAG) pipeline.

Demonstrates: chunking, TF-IDF retrieval (swap for dense embeddings
in production), BM25 vs dense retrieval, reranking, and Recall@K
evaluation. Does not require an LLM API key — the retrieval stage
is fully evaluated independently.

Real lesson: retrieval quality determines answer quality. A weak
retriever with a strong LLM still fails. Evaluate retrieval
(Recall@K, MRR) before evaluating the end-to-end system.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Knowledge base ────────────────────────────────────────────────────────────
DOCUMENTS = {
    "doc_eval":    """
Model evaluation: PR-AUC is preferred over ROC-AUC for imbalanced datasets.
Calibration measures whether predicted probabilities match observed frequencies.
The confusion matrix shows true positives, false positives, true negatives, and
false negatives. Precision-recall curves plot precision against recall at all
thresholds. Cost-sensitive thresholds set the decision boundary to minimise
the total expected cost of false positives and false negatives.
""",
    "doc_ab":      """
A/B testing: statistical power determines the minimum sample size needed to
detect a given effect size with a specified false-positive rate. The peeking
problem occurs when analysts check results before the pre-specified sample size
is reached, inflating the false-positive rate. CUPED reduces variance in A/B
tests by regressing out pre-experiment covariates. Sequential testing with
mSPRT allows early stopping without inflating false-positive rates.
""",
    "doc_churn":   """
Churn prediction and uplift modelling: target persuadables not risk.
Uplift modelling estimates the causal effect of a treatment on each customer.
Persuadables churn without intervention but stay with one.
Sure things do not churn regardless. Lost causes churn regardless of intervention.
The T-learner estimates the CATE by training separate models on treatment and
control groups and subtracting predictions.
""",
    "doc_survival":"""
Survival analysis handles time-to-event data with censoring. Kaplan-Meier
estimates survival probabilities non-parametrically. Cox proportional hazards
models estimate hazard ratios for covariates without specifying the baseline hazard.
Right-censored observations are customers who have not yet experienced the event.
The log-rank test compares survival curves between groups.
""",
    "doc_feature": """
Feature engineering and data leakage: preprocessing transforms must be fit only
on training data and applied to test data, never the reverse. StandardScaler,
TF-IDF vectorisers, and target encoders must live inside a Pipeline so they are
correctly fitted per cross-validation fold. The shuffle-label leakage guard checks
whether a model trained on permuted labels achieves near-chance performance.
""",
    "doc_sql":     """
SQL and data engineering: window functions compute aggregates over a partition
without collapsing rows. The fan-out grain bug occurs when joining tables at
different granularities, causing metrics to be double-counted. Common table
expressions (CTEs) improve readability of complex queries. Materialised views
cache expensive aggregations. Slowly changing dimensions (SCD Type 2) track
historical state in dimensional models.
""",
    "doc_rl":      """
Reinforcement learning and bandits: the explore-exploit trade-off determines
how much exploration to allocate versus exploiting the currently best known arm.
Thompson sampling maintains a Beta distribution posterior per arm and achieves
sub-linear regret. UCB1 uses an upper confidence bound bonus to balance
exploration. Contextual bandits condition arm selection on observable features.
LinUCB assumes linear reward in context and maintains per-arm ridge regression.
""",
    "doc_fairness":"""
Responsible AI and fairness: disparate impact is measured by the 80 percent rule
where the selection rate for a protected group must be at least 80 percent of
the majority group rate. Equalised odds requires equal true positive rates and
false positive rates across groups. Post-processing threshold adjustment can
repair equalised odds without retraining. Model cards document intended use,
limitations, and fairness evaluation results.
""",
}

# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_document(doc_id: str, text: str, chunk_size: int = 2,
                   overlap: int = 1) -> list[dict]:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = sentences[i:i + chunk_size]
        chunks.append({
            "id":      f"{doc_id}_c{len(chunks)}",
            "doc_id":  doc_id,
            "text":    " ".join(chunk_sents),
            "start":   i,
        })
        i += max(1, chunk_size - overlap)
    return chunks

all_chunks: list[dict] = []
for doc_id, text in DOCUMENTS.items():
    all_chunks.extend(chunk_document(doc_id, text, chunk_size=2, overlap=1))

print(f"Corpus: {len(DOCUMENTS)} documents -> {len(all_chunks)} chunks\n")

# ── Retrieval: TF-IDF (sparse) ────────────────────────────────────────────────
chunk_texts = [c["text"] for c in all_chunks]
tfidf_vec   = TfidfVectorizer(ngram_range=(1, 2), max_features=5000).fit(chunk_texts)
chunk_vecs  = tfidf_vec.transform(chunk_texts)

def retrieve_tfidf(query: str, k: int = 3) -> list[dict]:
    q_vec = tfidf_vec.transform([query])
    sims  = cosine_similarity(q_vec, chunk_vecs)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    return [(all_chunks[i], float(sims[i])) for i in top_k]

# ── Evaluation: Recall@K ──────────────────────────────────────────────────────
# Gold standard: (query, relevant_doc_id)
EVAL_SET = [
    ("which metric for fraud detection imbalanced data",    "doc_eval"),
    ("how to avoid peeking in experiments",                 "doc_ab"),
    ("target persuadables not churners",                    "doc_churn"),
    ("censored observations in survival models",            "doc_survival"),
    ("data leakage preprocessing pipeline",                 "doc_feature"),
    ("fan-out bug joining tables",                          "doc_sql"),
    ("explore exploit thompson sampling",                   "doc_rl"),
    ("disparate impact 80 percent rule",                    "doc_fairness"),
    ("CUPED variance reduction AB test",                    "doc_ab"),
    ("calibration predicted probabilities",                 "doc_eval"),
]

def recall_at_k(k: int) -> float:
    hits = 0
    for query, relevant_doc in EVAL_SET:
        results = retrieve_tfidf(query, k=k)
        retrieved_docs = {r["doc_id"] for r, _ in results}
        if relevant_doc in retrieved_docs:
            hits += 1
    return hits / len(EVAL_SET)

def mrr() -> float:
    rr_sum = 0.0
    for query, relevant_doc in EVAL_SET:
        results = retrieve_tfidf(query, k=10)
        for rank, (chunk, _) in enumerate(results, 1):
            if chunk["doc_id"] == relevant_doc:
                rr_sum += 1 / rank
                break
    return rr_sum / len(EVAL_SET)

print("-" * 45)
print("Retrieval evaluation (TF-IDF baseline)")
print("-" * 45)
for k in [1, 3, 5]:
    print(f"  Recall@{k}: {recall_at_k(k):.2f}")
print(f"  MRR:      {mrr():.3f}")
print()

# ── Sample queries ────────────────────────────────────────────────────────────
DEMO_QUERIES = [
    "how to reduce variance in an A/B test?",
    "what is the right metric for an imbalanced classifier?",
    "how does Thompson sampling work?",
]

print("-" * 60)
print("Demo: top-3 retrieved chunks per query")
print("-" * 60)
for query in DEMO_QUERIES:
    print(f"\nQuery: {query!r}")
    results = retrieve_tfidf(query, k=3)
    for rank, (chunk, score) in enumerate(results, 1):
        snippet = chunk["text"][:100].replace("\n", " ")
        print(f"  {rank}. [{score:.3f}] [{chunk['doc_id']}] {snippet}...")

# ── Naive vs retrieval-augmented comparison ───────────────────────────────────
print("\n" + "-" * 60)
print("Key insight: retrieval quality bounds RAG quality")
print("-" * 60)
print(f"If Recall@3 = {recall_at_k(3):.0%}: the LLM has the right context "
      f"{recall_at_k(3):.0%} of the time")
print("If Recall@3 = 50%: no matter how good the LLM, half the answers lack context")
print()
print("Production improvements over TF-IDF baseline:")
print("  1. Dense embeddings (sentence-transformers) - semantic recall")
print("  2. Hybrid BM25 + dense (combine sparse + semantic signals)")
print("  3. Cross-encoder reranking - slower but more precise top-K")
print("  4. Metadata filtering - filter by date/source before ANN search")
