# One-command reproducibility. `make help` lists targets.
.DEFAULT_GOAL := help
PY := python

.PHONY: help setup lint test projects case-study reproduce clean
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

case-study: ## Run the end-to-end churn + uplift case study
	cd case_study_churn_uplift && $(PY) run.py

reproduce: lint test projects case-study ## Full pipeline: lint -> test -> all projects
	@echo "All checks passed and every result reproduced from source."

clean: ## Remove generated figures and caches
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.pytest_cache' -type d -prune -exec rm -rf {} +
