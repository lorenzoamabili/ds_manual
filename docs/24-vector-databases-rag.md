# Vector Databases & Retrieval-Augmented Generation

Dense vector search is the foundational primitive behind semantic search, RAG,
duplicate detection, and recommendation. Understanding embeddings + approximate
nearest-neighbour (ANN) search is now a core DS competency.

---

## Embeddings

An embedding maps a high-dimensional object (text, image, user) to a dense
low-dimensional vector where **distance ≈ semantic similarity**.

| Source | Common model | Dimension |
|---|---|---|
| Sentence / paragraph | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| Long document | `text-embedding-3-large` (OpenAI) | 3072 |
| Image | CLIP ViT-B/32 | 512 |
| User behaviour | learned from interaction matrix | 64–256 |

**Cosine similarity** is the standard distance metric for text embeddings
(embeddings are L2-normalised, so cosine = dot product).

---

## Approximate nearest-neighbour search

Exact NN search is O(N·D) — prohibitive at millions of vectors. ANN trades
a small accuracy loss for sub-linear query time.

### FAISS (Facebook AI Similarity Search)

The most widely used ANN library. Key index types:

| Index | Speed | Recall | Memory | Use case |
|---|---|---|---|---|
| `IndexFlatL2` | Slowest | 100% | High | Ground truth / small N |
| `IndexIVFFlat` | Fast | ~95% | Medium | Millions of vectors |
| `IndexHNSWFlat` | Very fast | ~98% | High | Low-latency serving |
| `IndexIVFPQ` | Fastest | ~90% | Very low | Billions of vectors |

### Python example — FAISS semantic search

```python
import numpy as np

# pip install faiss-cpu sentence-transformers
# Demonstrating the concept with random vectors if packages not available
try:
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Corpus of documents
    docs = [
        "machine learning for fraud detection",
        "credit card anomaly detection with neural networks",
        "natural language processing for customer support",
        "transformer models for text classification",
        "time series forecasting with gradient boosting",
        "survival analysis for customer churn prediction",
        "reinforcement learning for recommendation systems",
        "A/B testing and causal inference in product analytics",
    ]

    # Encode documents and index
    doc_embeddings = model.encode(docs, normalize_embeddings=True)
    D = doc_embeddings.shape[1]

    index = faiss.IndexFlatIP(D)   # inner product = cosine on normalised vectors
    index.add(doc_embeddings.astype(np.float32))

    # Query
    queries = [
        "detecting financial fraud",
        "how to run experiments on a product",
    ]
    query_embeddings = model.encode(queries, normalize_embeddings=True)
    scores, indices = index.search(query_embeddings.astype(np.float32), k=3)

    for q, (score_row, idx_row) in zip(queries, zip(scores, indices)):
        print(f"\nQuery: {q!r}")
        for rank, (s, i) in enumerate(zip(score_row, idx_row)):
            print(f"  {rank+1}. [{s:.3f}] {docs[i]}")

except ImportError:
    # Fallback: simulate with random embeddings
    print("sentence-transformers / faiss not installed — showing concept with random vectors")
    rng = np.random.default_rng(42)
    D, N = 64, 100
    docs_fake = [f"document_{i}" for i in range(N)]
    vecs = rng.normal(0, 1, (N, D)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)   # L2-normalise

    query = rng.normal(0, 1, (1, D)).astype(np.float32)
    query /= np.linalg.norm(query)

    sims = vecs @ query.T
    top3 = np.argsort(sims[:,0])[-3:][::-1]
    print("Top-3 results by cosine similarity:", top3, "scores:", sims[top3, 0].round(3))
```

---

## Retrieval-Augmented Generation (RAG)

RAG augments an LLM with a retrieval step: instead of relying on parametric
memory (weights), the model retrieves relevant context from an external store
at inference time. This fixes three core LLM problems:

1. **Knowledge cutoff** — retrieved docs can be current
2. **Hallucination** — the model is grounded in retrieved text
3. **Privacy** — sensitive data never enters the LLM's training

### RAG pipeline

```
Query → [Embed query] → [ANN search] → [Retrieved chunks] → [LLM prompt] → Answer
```

```python
import textwrap

# Minimal RAG pipeline using TF-IDF retrieval + a mock LLM step
# (replace TF-IDF with dense embeddings + FAISS in production)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    "PR-AUC is preferred over ROC-AUC for imbalanced datasets because it focuses "
    "on the minority class performance.",

    "CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance in "
    "A/B tests by regressing out pre-experiment covariates correlated with the metric.",

    "The fan-out grain bug occurs when joining tables at different granularities, "
    "causing metric values to be counted multiple times.",

    "Uplift modelling targets the persuadables — customers who would churn without "
    "an intervention but stay with one — rather than the highest-risk customers.",

    "Kaplan-Meier curves estimate survival probabilities non-parametrically and "
    "handle right-censored observations correctly.",

    "The Cox proportional hazards model estimates the effect of covariates on "
    "the hazard rate while leaving the baseline hazard unspecified.",

    "Disparate impact is measured by the 80% rule: the selection rate for a "
    "protected group must be at least 80% of the majority group's rate.",

    "Calibration measures whether predicted probabilities match observed frequencies. "
    "A model predicting 0.7 for an event should be right ~70% of the time.",
]

# ── Retriever ──────────────────────────────────────────────────────────────────
vec = TfidfVectorizer().fit(KNOWLEDGE_BASE)
doc_vecs = vec.transform(KNOWLEDGE_BASE)

def retrieve(query: str, k: int = 2) -> list[str]:
    q_vec = vec.transform([query])
    sims  = cosine_similarity(q_vec, doc_vecs)[0]
    top_k = np.argsort(sims)[-k:][::-1]
    return [KNOWLEDGE_BASE[i] for i in top_k]

# ── Mock LLM step ─────────────────────────────────────────────────────────────
def rag_answer(query: str) -> None:
    context = retrieve(query, k=2)
    prompt  = f"""Answer the question using only the provided context.

Context:
{chr(10).join(f'- {c}' for c in context)}

Question: {query}
Answer:"""
    print("=" * 60)
    print(f"Query: {query}")
    print("\nRetrieved context:")
    for c in context:
        print(f"  • {textwrap.shorten(c, 80)}")
    print("\n[LLM would generate answer from this prompt — returning prompt for inspection]")
    print(textwrap.indent(prompt, "  "))

rag_answer("Which metric should I use for fraud detection models?")
rag_answer("How do I reduce variance in my A/B test?")
```

---

## Chunking strategies

How you split documents before embedding determines retrieval quality.

| Strategy | When to use |
|---|---|
| Fixed-size (512 tokens, 50-token overlap) | Default; works for most prose |
| Sentence-aware | When paragraph boundaries matter |
| Semantic chunking | Split where embedding similarity drops |
| Parent-child retrieval | Return larger parent chunk after matching small child |

**The 512-token rule**: most embedding models are trained with sequences ≤ 512
tokens. Chunks longer than this degrade embedding quality.

---

## Evaluation

### Retrieval metrics

- **Recall@K**: fraction of relevant documents in top-K results
- **MRR** (Mean Reciprocal Rank): position of first relevant result
- **NDCG@K**: graded relevance, discounted by rank

### End-to-end RAG metrics

- **Faithfulness**: does the answer contain only claims supported by retrieved context?
- **Answer relevance**: does the answer address the question?
- **Context precision**: how much of the retrieved context was actually used?

Frameworks: [RAGAS](https://github.com/explodinggradients/ragas),
[TruLens](https://github.com/truera/trulens).

---

## Production RAG architecture

```
                  ┌─────────────────────────────────┐
  Document        │   Ingestion pipeline             │
  sources ──────► │   chunk → embed → upsert        │
                  └──────────────┬──────────────────┘
                                 │
                         Vector DB (Pinecone,
                         Qdrant, pgvector, FAISS)
                                 │
  User query ──► embed ──► ANN search ──► retrieved chunks
                                              │
                                    ┌─────────▼────────┐
                                    │  LLM (GPT-4,     │
                                    │  Claude, Llama)  │
                                    └─────────┬────────┘
                                              │
                                           Answer
```

**Key production decisions:**

- **Hybrid search**: combine dense (embedding) + sparse (BM25/TF-IDF) search.
  Dense retrieval misses exact keyword matches; sparse misses paraphrases.
- **Reranking**: a cross-encoder reranker (slower but more accurate) rescores
  the top-50 ANN results before returning top-5 to the LLM.
- **Metadata filtering**: filter by date, source, document type before ANN
  search to reduce irrelevant context.
- **Context window budget**: modern LLMs support 128K+ token contexts, but
  LLM attention degrades for content in the middle ("lost in the middle").
  Prefer shorter, higher-precision context over long, noisy context.

---

## When to use RAG vs fine-tuning

| Need | Solution |
|---|---|
| Current knowledge (post-training cutoff) | RAG |
| Domain-specific style / format | Fine-tuning |
| Private/proprietary data (security) | RAG (data never in weights) |
| Improved task performance (classification) | Fine-tuning |
| Grounded, verifiable answers with citations | RAG |
| Low latency (no retrieval step) | Fine-tuning |

In practice: RAG for knowledge, fine-tuning for behaviour.
