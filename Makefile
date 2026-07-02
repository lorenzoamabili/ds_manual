# One-command reproducibility. `make help` lists targets.
.DEFAULT_GOAL := help
PY := python

.PHONY: help setup lint test projects case-study reproduce notebooks clean
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest ruff

lint: ## Static checks
	ruff check src tests

test: ## Run the unit tests
	$(PY) -m pytest -q

projects: ## Run all projects (writes metrics + figures)
	cd projects/p1_supervised_learning     && $(PY) train.py
	cd projects/p2_time_series_forecasting && $(PY) forecast.py
	cd projects/p3_causal_inference        && $(PY) causal.py
	cd projects/p4_unsupervised_learning   && $(PY) cluster.py
	cd projects/p5_survival_analysis       && $(PY) survival.py
	cd projects/p6_fairness_audit          && $(PY) fairness.py
	cd projects/p7_anomaly_detection       && $(PY) detect.py
	cd projects/p8_recommender             && $(PY) recommend.py
	cd projects/p9_nlp_classification      && $(PY) classify.py
	cd projects/p10_optimization           && $(PY) optimize.py

case-study: ## Run the end-to-end churn + uplift case study
	cd case_study_churn_uplift && $(PY) run.py

notebooks: ## Execute all Jupyter notebooks top-to-bottom (requires jupyter)
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
	    projects/p1_supervised_learning/notebook.ipynb \
	    projects/p2_time_series_forecasting/notebook.ipynb \
	    projects/p3_causal_inference/notebook.ipynb \
	    projects/p4_unsupervised_learning/notebook.ipynb \
	    projects/p5_survival_analysis/notebook.ipynb \
	    projects/p6_fairness_audit/notebook.ipynb \
	    projects/p7_anomaly_detection/notebook.ipynb \
	    projects/p8_recommender/notebook.ipynb \
	    projects/p9_nlp_classification/notebook.ipynb \
	    projects/p10_optimization/notebook.ipynb \
	    case_study_churn_uplift/notebook.ipynb \
	    notebooks/01-statistics-fundamentals.ipynb \
	    notebooks/02-model-evaluation.ipynb \
	    notebooks/03-ab-testing.ipynb \
	    notebooks/04-nlp-text-classification.ipynb \
	    notebooks/05-feature-engineering.ipynb

reproduce: lint test projects case-study ## Full pipeline: lint -> test -> all projects
	@echo "All checks passed and every result reproduced from source."

clean: ## Remove generated figures and caches
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
