"""A behavioural test for the practice the manual cares about most: no leakage.

Rather than test a metric value (which can drift with library versions), we test
the *structural* property that prevents leakage — that preprocessing lives inside
the cross-validated estimator, so it is only ever fit on training folds.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000))])


def test_preprocessing_is_inside_the_estimator():
    """The scaler must be a step of the pipeline, not applied beforehand."""
    pipe = build_pipeline()
    step_names = [name for name, _ in pipe.steps]
    assert "scale" in step_names
    assert step_names.index("scale") < step_names.index("clf")


def test_pipeline_cross_validates_without_leakage_and_performs():
    """cross_val_score refits the scaler per fold; a leak-free model still works."""
    X, y = load_breast_cancer(return_X_y=True)
    scores = cross_val_score(build_pipeline(), X, y, cv=5, scoring="roc_auc")
    assert len(scores) == 5
    assert scores.mean() > 0.95          # strong but honest, no leakage inflating it


def test_shuffled_labels_collapse_to_chance():
    """The canonical leakage sanity check: random labels -> ~chance performance."""
    X, y = load_breast_cancer(return_X_y=True)
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y)
    scores = cross_val_score(build_pipeline(), X, y_shuf, cv=5, scoring="roc_auc")
    assert scores.mean() < 0.65          # if this were high, we'd have leakage
