# HRTech Case Studies

## Google — Project Oxygen and the "predictive manager" problem

**Problem:** Google tried to determine whether managers matter. Their null hypothesis (which they expected to confirm) was that technical excellence is what drives team performance, and management is noise. If true, eliminating the management layer would improve efficiency. Their data contradicted the hypothesis.

**Approach:** People Analytics team analysed performance review scores, employee satisfaction survey data (Googlegeist), turnover rates, and promotion data at the team level. They used a combination of regression (manager behaviour → team outcomes) and **uplift modelling** to identify *which specific management behaviours* were causally associated with better team outcomes.

**Key technical decisions:**
- **Text analysis of 360° review comments**: unstructured feedback was the richest signal. NLP (initially keyword-based, later topic modelling and sentiment) extracted recurring themes from thousands of comments. This surfaced eight behaviours that differentiated high- and low-rated managers — data-driven rather than assumption-driven.
- **Causal problem**: manager assignment is not random. Good managers may select better teams, or better teams may attract good managers. The team used **fixed-effects models** (controlling for team × time intercepts) and longitudinal analysis (same team, new manager) to partially address confounding.
- **Selection on observables vs. experiment**: Google ultimately ran a field experiment — they trained a randomly selected set of managers on the identified behaviours and measured impact on their teams. This is the only way to confirm the causal link.

**What happened:** They found that managers *do* matter — specifically on 8 behaviours (Project Oxygen behaviours). The top factor was not technical skill but being a "good coach." The bottom factor (8th) was technical skill. This reversed the prior and fundamentally changed how Google hired and trained managers.

**Transferable lesson:** People analytics done right is a **causal inference problem**, not a correlation analysis. Finding that good managers correlate with better teams tells you nothing about whether *improving* managers would improve teams. Design experiments or use quasi-experimental methods. See [doc 09](../09-causal-inference-and-experimentation.md).

---

## IBM — employee attrition prediction and the Goodhart trap

**Problem:** IBM's HR team built an attrition prediction model (publicised extensively, including releasing the HR Analytics dataset on Kaggle). The model predicts, for each employee, the probability of leaving within the next year. The intended use: identify at-risk employees and offer retention interventions.

**Approach:** Gradient boosting classifier on HR features: tenure, job level, satisfaction scores (survey-derived), overtime hours, distance from home, recent promotions, performance rating. AUC ≈ 0.84 on the IBM dataset.

**Key technical decisions and failures:**

- **Goodhart's Law**: when the model's score is used operationally (managers see who is "high risk" and intervene), the score becomes a target and stops being a good measure. Employees who learn they are flagged as at-risk may leave *because* of the stigma of being flagged. Employees who are not flagged but are actually planning to leave are never intervened on. The model changes the base rates it was trained on.
- **Privacy and trust**: the IBM HR dataset is synthetic, but real employee attrition scoring involves sensitive personal data (satisfaction surveys, performance ratings, absenteeism). Employees who discover they are being scored for flight risk may feel surveilled and their trust in the organisation decreases — ironically increasing attrition.
- **Protected characteristics**: features correlated with protected characteristics (family status, distance from home, overtime) can result in disparate impact — women with caregiving responsibilities may score higher flight risk not because they're less engaged but because of life circumstances. See [doc 19](../19-responsible-ai-and-fairness.md).

**Transferable lesson:** Predictions that inform interventions change the very behaviour they predict. This **prediction → intervention → behaviour change → new labels** loop is the defining challenge of HR ML. The ethical and Goodhart problems are not edge cases — they are intrinsic to the use case. Attrition scoring requires an ethics review, not just an AUC check.

---

## LinkedIn — Economic Graph and labour market analytics

**Problem:** LinkedIn's Economic Graph is the data product underlying LinkedIn Insights, Salary Insights, and the LinkedIn Workforce Report. The goal: understand labour market dynamics in real time — skills demand, hiring velocity, talent migration — using LinkedIn's dataset of 900M+ profiles and 60M+ jobs.

**Approach:**

- **Skills taxonomy**: LinkedIn maintains a 36,000-entity skills taxonomy built by clustering skill mentions from profiles and job postings using NLP. New skills are detected by monitoring velocity of new skill terms and clustering them with existing skills. See [doc 10](../10-nlp-and-llms.md).
- **Hiring rate estimation**: LinkedIn estimates job vacancy and hiring rates from job posting lifecycles (posted → filled, estimated from posting removal time) and member activity (profile updates, job title changes). This requires correcting for platform engagement bias — companies that post more jobs on LinkedIn aren't necessarily representative of all employers.
- **Talent flow analysis**: aggregate migration of talent between companies, regions, and industries is a graph analysis problem. LinkedIn uses graph algorithms on a bipartite member × company graph to compute flow rates and identify talent hubs. See [doc 21](../21-graph-and-network-analysis.md).

**Key challenge — representativeness bias:** LinkedIn's 900M members are not a representative sample of the global workforce. High-income, knowledge-worker, English-speaking workers are dramatically overrepresented. Any labour market statistic derived from LinkedIn data has this selection bias. LinkedIn addresses this with post-stratification weighting using external labour force surveys (BLS in the US, Eurostat in Europe) as calibration targets.

**Transferable lesson:** Platform data is a biased sample of the population it claims to represent. Any insight derived from platform data must be interpreted through the lens of *who is on the platform and why*. Post-stratification to external benchmarks is the standard correction, but it requires access to those benchmarks.

---

## Cross-cutting lessons

1. **Causal inference, not correlation**, in people analytics. "Good managers correlate with good teams" is uninformative without an experiment or quasi-experimental design.
2. **Goodhart's Law** is endemic to HR prediction: models that score employees change how employees and managers behave, invalidating the model's training distribution.
3. **Ethics review required** for HR ML: attrition scoring, performance prediction, and hiring screening all involve protected characteristics and power asymmetries.
4. **Platform representativeness bias**: any analysis on HR platform data (LinkedIn, Glassdoor, Indeed) is biased toward that platform's user demographics and must be calibrated to external benchmarks.
