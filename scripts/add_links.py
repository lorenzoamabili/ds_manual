"""
Add Wikipedia-style inline links to all markdown docs.

Rules:
  - Link first occurrence of each term per document
  - Never link inside code blocks or inline code
  - Never link inside an existing [...](...)
  - Never self-link
  - Case-insensitive matching, preserve original capitalisation
  - Replacements applied right-to-left to avoid offset drift
"""
import re
from pathlib import Path

DOCS = Path(__file__).parent.parent / "docs"

# Term pattern → target doc filename (relative to docs/)
LINKS: dict[str, str] = {
    # 01
    r"reproducib(?:ility|le)":                "01-lifecycle-and-reproducibility.md",
    # 02
    r"p-value":                               "02-statistics-that-matter.md",
    r"bootstrap(?:ping)?":                    "02-statistics-that-matter.md",
    r"statistical power":                     "02-statistics-that-matter.md",
    r"central limit theorem":                 "02-statistics-that-matter.md",
    r"Benjamini.Hochberg":                    "02-statistics-that-matter.md",
    # 03
    r"(?:data |target |temporal )?leakage":   "03-data-and-feature-engineering.md",
    r"feature engineer(?:ing|ed)?":           "03-data-and-feature-engineering.md",
    r"imputation":                            "03-data-and-feature-engineering.md",
    r"missingness":                           "03-data-and-feature-engineering.md",
    r"one-hot encod(?:ing|ed)?":              "03-data-and-feature-engineering.md",
    r"target encod(?:ing|ed)?":               "03-data-and-feature-engineering.md",
    # 04
    r"ROC-AUC":                               "04-evaluation-and-validation.md",
    r"PR-AUC":                                "04-evaluation-and-validation.md",
    r"calibrat(?:ion|ed|e)":                 "04-evaluation-and-validation.md",
    r"cross-validat(?:ion|ed|e)":            "04-evaluation-and-validation.md",
    r"stratified k-fold":                     "04-evaluation-and-validation.md",
    r"precision-recall":                      "04-evaluation-and-validation.md",
    r"confusion matrix":                      "04-evaluation-and-validation.md",
    # 05
    r"logistic regression":                   "05-supervised-learning.md",
    r"gradient boost(?:ing|ed)?":            "05-supervised-learning.md",
    r"XGBoost":                               "05-supervised-learning.md",
    r"LightGBM":                              "05-supervised-learning.md",
    r"random forest":                         "05-supervised-learning.md",
    r"SHAP(?: values?)?":                     "05-supervised-learning.md",
    r"permutation importance":                "05-supervised-learning.md",
    r"class imbalance":                       "05-supervised-learning.md",
    r"regularisa?tion":                       "05-supervised-learning.md",
    # 06
    r"[Kk]-?[Mm]eans":                       "06-unsupervised-learning.md",
    r"\bPCA\b":                               "06-unsupervised-learning.md",
    r"silhouette(?: score)?":                "06-unsupervised-learning.md",
    r"cluster(?:ing|ed)?":                   "06-unsupervised-learning.md",
    r"dimensionality reduction":              "06-unsupervised-learning.md",
    r"\bUMAP\b":                              "06-unsupervised-learning.md",
    r"\bt-SNE\b":                             "06-unsupervised-learning.md",
    # 07
    r"time-series(?: model(?:ling|ing)?)?":  "07-time-series-forecasting.md",
    r"forecast(?:ing|ed|s)?":               "07-time-series-forecasting.md",
    r"\bARIMA\b":                             "07-time-series-forecasting.md",
    r"\bETS\b":                               "07-time-series-forecasting.md",
    r"rolling-origin":                        "07-time-series-forecasting.md",
    r"seasonalit(?:y|ies)":                  "07-time-series-forecasting.md",
    r"quantile regression":                   "07-time-series-forecasting.md",
    r"\bDeepAR\b":                            "07-time-series-forecasting.md",
    # 08
    r"collaborative filtering":               "08-recommendation-systems.md",
    r"matrix factori[sz]ation":              "08-recommendation-systems.md",
    r"\bSVD\b":                               "08-recommendation-systems.md",
    r"popularity bias":                       "08-recommendation-systems.md",
    # 09
    r"A/B test(?:ing|s|ed)?":               "09-causal-inference-and-experimentation.md",
    r"causal inference":                      "09-causal-inference-and-experimentation.md",
    r"propensity(?: score)?":               "09-causal-inference-and-experimentation.md",
    r"confound(?:ing|er|ed)?":              "09-causal-inference-and-experimentation.md",
    r"randomis(?:ed|ation) controlled trial": "09-causal-inference-and-experimentation.md",
    r"difference-in-differences":             "09-causal-inference-and-experimentation.md",
    r"instrumental variable":                 "09-causal-inference-and-experimentation.md",
    r"average treatment effect":              "09-causal-inference-and-experimentation.md",
    r"doubly-robust":                         "09-causal-inference-and-experimentation.md",
    r"\bSUTVA\b":                             "09-causal-inference-and-experimentation.md",
    r"Simpson's paradox":                     "09-causal-inference-and-experimentation.md",
    # 10
    r"TF-IDF":                                "10-nlp-and-llms.md",
    r"transformer(?: model)?":               "10-nlp-and-llms.md",
    r"\bBERT\b":                              "10-nlp-and-llms.md",
    r"named entity recognition":             "10-nlp-and-llms.md",
    r"\bLLM\b":                               "10-nlp-and-llms.md",
    r"word embed(?:ding|ded)?":              "10-nlp-and-llms.md",
    r"topic model(?:ling|ing)?":            "10-nlp-and-llms.md",
    r"sentiment analysis":                    "10-nlp-and-llms.md",
    # 11
    r"transfer learning":                     "11-computer-vision.md",
    r"convolutional neural network":          "11-computer-vision.md",
    r"\bCNN\b":                               "11-computer-vision.md",
    r"\bResNet\b":                            "11-computer-vision.md",
    r"fine-tun(?:ing|ed|e)":                "11-computer-vision.md",
    r"data augmentation":                     "11-computer-vision.md",
    r"object detection":                      "11-computer-vision.md",
    # 12
    r"linear programming":                    "12-optimization.md",
    r"\bMILP\b":                              "12-optimization.md",
    r"integer programming":                   "12-optimization.md",
    r"vehicle routing":                       "12-optimization.md",
    r"knapsack":                              "12-optimization.md",
    # 13
    r"Isolation Forest":                      "13-anomaly-detection.md",
    r"anomaly detection":                     "13-anomaly-detection.md",
    r"Local Outlier Factor":                 "13-anomaly-detection.md",
    # 14
    r"(?:data |concept |model )?drift":      "14-mlops-and-productionization.md",
    r"\bMLOps\b":                             "14-mlops-and-productionization.md",
    r"shadow deploy(?:ment)?":               "14-mlops-and-productionization.md",
    r"canary deploy(?:ment)?":               "14-mlops-and-productionization.md",
    r"model registry":                        "14-mlops-and-productionization.md",
    r"experiment tracking":                   "14-mlops-and-productionization.md",
    r"training-serving skew":                 "14-mlops-and-productionization.md",
    # 16
    r"survival analysis":                     "16-survival-analysis.md",
    r"Kaplan-Meier":                          "16-survival-analysis.md",
    r"Cox proportional hazards":              "16-survival-analysis.md",
    r"hazard ratio":                          "16-survival-analysis.md",
    r"(?:right-)?censor(?:ing|ed)?":        "16-survival-analysis.md",
    r"time-to-event":                         "16-survival-analysis.md",
    # 17
    r"\bCUPED\b":                             "17-experimentation-advanced.md",
    r"sequential testing":                    "17-experimentation-advanced.md",
    r"multi-armed bandit":                    "17-experimentation-advanced.md",
    r"switchback design":                     "17-experimentation-advanced.md",
    r"novelty effect":                        "17-experimentation-advanced.md",
    # 18
    r"window function":                       "18-sql-and-data-engineering.md",
    r"fan-out(?: bug| grain)?":              "18-sql-and-data-engineering.md",
    # 19
    r"disparate impact":                      "19-responsible-ai-and-fairness.md",
    r"equalised odds":                        "19-responsible-ai-and-fairness.md",
    r"demographic parity":                    "19-responsible-ai-and-fairness.md",
    r"responsible AI":                        "19-responsible-ai-and-fairness.md",
    # 20
    r"Bayesian(?: inference| updating)?":    "20-bayesian-and-probabilistic.md",
    r"Beta-Binomial":                         "20-bayesian-and-probabilistic.md",
    r"posterior(?: distribution)?":          "20-bayesian-and-probabilistic.md",
    r"conjugate prior":                       "20-bayesian-and-probabilistic.md",
    r"credible interval":                     "20-bayesian-and-probabilistic.md",
    # 21
    r"graph analysis":                        "21-graph-and-network-analysis.md",
    r"\bNetworkX\b":                          "21-graph-and-network-analysis.md",
    r"community detection":                   "21-graph-and-network-analysis.md",
    r"betweenness centrality":               "21-graph-and-network-analysis.md",
    r"\bLouvain\b":                           "21-graph-and-network-analysis.md",
    r"bipartite(?: graph)?":                "21-graph-and-network-analysis.md",
    # 22
    r"geospatial":                            "22-geospatial.md",
    r"\bH3\b":                                "22-geospatial.md",
    r"spatial autocorrelation":              "22-geospatial.md",
    r"Moran's I":                             "22-geospatial.md",
    r"hexagonal grid":                        "22-geospatial.md",
}

# Sort longest pattern first so longer matches take priority
SORTED_LINKS = sorted(LINKS.items(), key=lambda kv: len(kv[0]), reverse=True)


def get_protected_ranges(text: str) -> list[tuple[int, int]]:
    """
    Return list of (start, end) ranges that must not be modified:
      - fenced code blocks  ```...```
      - inline code         `...`
      - existing links      [...](...)  — both the text and URL parts
    """
    ranges = []
    # Fenced + inline code
    for m in re.finditer(r'```[\s\S]*?```|`[^`\n]+`', text):
        ranges.append((m.start(), m.end()))
    # Existing markdown links — protect the whole [text](url) span
    for m in re.finditer(r'\[[^\]]*\]\([^)]*\)', text):
        ranges.append((m.start(), m.end()))
    # Markdown image links
    for m in re.finditer(r'!\[[^\]]*\]\([^)]*\)', text):
        ranges.append((m.start(), m.end()))
    return ranges


def overlaps(start: int, end: int, protected: list[tuple[int, int]]) -> bool:
    return any(s < end and e > start for s, e in protected)


def process_doc(path: Path) -> int:
    text = path.read_text(encoding='utf-8')
    src_doc = path.name

    protected = get_protected_ranges(text)
    used: set[str] = set()

    # Collect all replacements: (start, end, new_text)
    replacements: list[tuple[int, int, str]] = []

    for pattern, target in SORTED_LINKS:
        if target == src_doc:
            continue
        if pattern in used:
            continue

        regex = re.compile(r'(?<!\[)(?<!\w)(' + pattern + r')(?!\w)(?!\])',
                           re.IGNORECASE)

        for m in regex.finditer(text):
            start, end = m.start(), m.end()
            if overlaps(start, end, protected):
                continue
            # Also skip if already queued for replacement (avoid double-linking)
            if any(s < end and e > start for s, e, _ in replacements):
                continue
            matched_text = m.group(1)
            replacements.append((start, end, f'[{matched_text}]({target})'))
            used.add(pattern)
            break  # only first occurrence

    if not replacements:
        return 0

    # Apply right-to-left so earlier positions stay valid
    replacements.sort(key=lambda r: r[0], reverse=True)
    chars = list(text)
    for start, end, new_text in replacements:
        chars[start:end] = list(new_text)

    new_text = ''.join(chars)
    path.write_text(new_text, encoding='utf-8')
    return len(replacements)


def main():
    total = 0
    doc_files = (
        sorted(DOCS.glob('*.md')) +
        sorted((DOCS / 'case_studies').glob('*.md'))
    )
    for path in doc_files:
        if path.name == 'index.md':
            continue
        n = process_doc(path)
        if n:
            print(f"  {path.name}: +{n} links")
            total += n
    print(f"\nTotal links added: {total}")


if __name__ == '__main__':
    main()
