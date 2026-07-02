"""
Project 9 — NLP Text Classification
======================================
Dataset : 20 Newsgroups (4-category subset, bundled with scikit-learn).
Goal    : Classify news articles using bag-of-words models; compare
          Naive Bayes, Logistic Regression (TF-IDF), and Linear SVM.

Real lesson: TF-IDF + logistic regression is a strong NLP baseline that
often beats Naive Bayes and rivals simple neural approaches. Don't reach
for transformers until you've beaten the linear baseline. Inspecting top
features per class reveals what the model actually learned.

Run:  python classify.py
Outputs: metrics.md, confusion_matrix.png, top_features.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

OUT = Path(__file__).parent

# ── Load 20 Newsgroups (4 categories) ────────────────────────────────────────
CATEGORIES = [
    "sci.med",
    "sci.space",
    "rec.sport.baseball",
    "talk.politics.guns",
]
print("Loading 20 Newsgroups (4 categories)…")
train_data = fetch_20newsgroups(subset="train", categories=CATEGORIES,
                                 remove=("headers", "footers", "quotes"))
test_data  = fetch_20newsgroups(subset="test",  categories=CATEGORIES,
                                 remove=("headers", "footers", "quotes"))

X_train, y_train = train_data.data, train_data.target
X_test,  y_test  = test_data.data,  test_data.target
class_names       = train_data.target_names

print(f"Train: {len(X_train):,} docs | Test: {len(X_test):,} docs")
print(f"Classes: {class_names}")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Naive Bayes (Count)": Pipeline([
        ("vec", CountVectorizer(max_features=30_000, ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.1)),
    ]),
    "Logistic Reg (TF-IDF)": Pipeline([
        ("vec", TfidfVectorizer(max_features=30_000, ngram_range=(1, 2),
                                sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ]),
    "Linear SVM (TF-IDF)": Pipeline([
        ("vec", TfidfVectorizer(max_features=30_000, ngram_range=(1, 2),
                                sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ]),
}

results = {}
preds_all = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    preds_all[name] = preds
    results[name] = {
        "Accuracy":  accuracy_score(y_test, preds),
        "F1-macro":  f1_score(y_test, preds, average="macro"),
        "F1-weighted": f1_score(y_test, preds, average="weighted"),
    }
    print(f"{name}: acc={results[name]['Accuracy']:.3f} | "
          f"F1-macro={results[name]['F1-macro']:.3f}")

df_results = pd.DataFrame(results).T
print("\n" + df_results.round(3).to_string())

# ── Write metrics.md ─────────────────────────────────────────────────────────
best_model = df_results["F1-macro"].idxmax()
with open(OUT / "metrics.md", "w") as f:
    f.write("# P9 · NLP Text Classification — Metrics\n\n")
    f.write("**Dataset:** 20 Newsgroups (4 categories: sci.med, sci.space, "
            "rec.sport.baseball, talk.politics.guns)\n\n")
    f.write(df_results.round(3).to_markdown())
    f.write(f"\n\n**Best model:** {best_model}\n\n")
    f.write("## Key insight\n\n")
    f.write("TF-IDF + linear models are a strong NLP baseline. ")
    f.write("They are fast, interpretable (inspect top coefficients), and often "
            "within 2–3 points of transformer models on topic classification. ")
    f.write("Don't jump to BERT until you've beaten this baseline.\n")

# ── Confusion matrix for best model ──────────────────────────────────────────
best_preds = preds_all[best_model]
cm = confusion_matrix(y_test, best_preds)
short_names = [c.split(".")[-1] for c in class_names]

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short_names, yticklabels=short_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix — {best_model}")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=120)
plt.close()

# ── Top features per class (LR) ───────────────────────────────────────────────
lr_pipe = models["Logistic Reg (TF-IDF)"]
lr_clf  = lr_pipe.named_steps["clf"]
vocab   = lr_pipe.named_steps["vec"].get_feature_names_out()
top_n   = 12

fig, axes = plt.subplots(1, len(class_names), figsize=(16, 5), sharey=False)
for ax, cls_idx, cls_name in zip(axes, range(len(class_names)), class_names):
    coefs     = lr_clf.coef_[cls_idx]
    top_pos   = np.argsort(coefs)[-top_n:]
    top_words = vocab[top_pos]
    top_coefs = coefs[top_pos]
    ax.barh(range(top_n), top_coefs, color="#3498db")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_words, fontsize=8)
    ax.set_title(cls_name.replace(".", "\n"), fontsize=9)
    ax.set_xlabel("Coefficient")
fig.suptitle(f"Top {top_n} features per class — Logistic Regression (TF-IDF)", fontsize=11)
plt.tight_layout()
plt.savefig(OUT / "top_features.png", dpi=120)
plt.close()

print("\nOutputs written: metrics.md, confusion_matrix.png, top_features.png")
print(f"\nLesson: '{best_model}' is the best linear baseline.")
print("Inspect top_features.png — the model learned sensible vocabulary per topic.")
print("Beat this before reaching for transformers.")
