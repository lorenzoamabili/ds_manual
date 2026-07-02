# HealthTech Case Studies

## Google Health — diabetic retinopathy screening

**Problem:** Diabetic retinopathy (DR) is the leading cause of preventable blindness globally. A trained ophthalmologist can diagnose it from a retinal fundus photograph. But there are not enough ophthalmologists in rural India, Thailand, and Sub-Saharan Africa — where diabetic patients have the fewest access points for screening.

**Approach:** Google trained a deep [CNN](../11-computer-vision.md) (InceptionV3 [fine-tuned](../11-computer-vision.md)) on 128,000 retinal images graded by a panel of US board-certified ophthalmologists. The model classifies severity into five grades and was validated on independent datasets from India and the US.

**Key technical decisions:**
- **Label aggregation**: each image was graded by 3-7 ophthalmologists. Disagreement was resolved by majority vote, but importantly the *distribution* of grades (not just the majority) was used to define uncertainty. Images with high inter-rater disagreement were flagged for model uncertainty, not forced into a single label.
- **Operating point selection**: the model was evaluated at a specific sensitivity threshold (≥90%) set by clinical requirement — missing a case of moderate DR is far more harmful than a false referral. This is a domain-driven threshold decision, not a default 0.5. See [doc 04](../04-evaluation-and-validation.md).
- **Subgroup evaluation**: the model was validated separately on different camera types, image quality grades, and patient demographics. A model that works well on high-quality images from US clinics may fail on lower-quality images from rural mobile screening units. See [doc 19](../19-responsible-ai-and-fairness.md).

**What failed first (in deployment):** The Thailand deployment revealed that real-world image quality was much lower than training data. The model's AUC dropped from 0.99 (on the curated validation set) to ~0.96 on in-the-field images. They retrained on locally collected images — a standard domain adaptation fix, but it required a field data collection pipeline they had not originally planned.

**Transferable lesson:** Validation performance on a curated test set is not the same as deployment performance. The gap between them is the *distribution shift* between data collection context and deployment context. Test on data collected in the *deployment* environment before claiming clinical readiness.

---

## NHS — 30-day hospital readmission prediction

**Problem:** The NHS (and US CMS) penalise hospitals for 30-day readmissions above a risk-adjusted baseline. The clinical goal: identify high-risk patients at discharge and provide targeted follow-up care (phone calls, care coordinators, GP handover).

**Approach:** A [gradient boosting](../05-supervised-learning.md) classifier trained on EHR data: admission diagnoses (ICD codes), comorbidities, lab values at discharge, length of stay, prior admission history, social deprivation index, discharge destination.

**Key technical decisions:**
- **LACE index as baseline**: before any ML, the LACE index (Length of stay, Acuity of admission, Comorbidity, Emergency department visits) is a well-validated clinical score. Any ML model must beat LACE on both discrimination (AUC) and [calibration](../04-evaluation-and-validation.md) to justify the added complexity. See [doc 04](../04-evaluation-and-validation.md).
- **Calibration is critical in clinical settings**: a nurse using the model to prioritise follow-up calls needs to know that a "70% readmission risk" means roughly 70 in 100 similar patients are readmitted — not a relative score. Miscalibrated models lead to incorrect resource allocation.
- **Protected characteristic audit**: readmission rates differ by socioeconomic deprivation, ethnicity, and geography. The model should predict readmission, not deprivation — and the two are correlated. A fairness audit (see [doc 19](../19-responsible-ai-and-fairness.md)) is mandatory before deployment in NHS settings.

**What failed first:** The first deployed model was evaluated on retrospective data from 2019-2020. COVID disrupted admission patterns, discharge pathways, and readmission rates so severely that the model's calibration collapsed in 2020-2021. The hospital had no monitoring pipeline and did not detect the [drift](../14-mlops-and-productionization.md) for months.

**Transferable lesson:** Clinical models *must* have real-time performance monitoring with drift detection. Patient populations, coding practices, and care pathways change — a model without monitoring is an untested model after its training cutoff. See [doc 14](../14-mlops-and-productionization.md).

---

## Sepsis early warning — the Epic deterioration index controversy

**Problem:** Sepsis kills ~250,000 people/year in the US. It is treatable if caught early. Epic Systems deployed an in-hospital sepsis early warning model (the "Deterioration Index", EDI) to hundreds of US hospitals, generating real-time alerts for at-risk patients.

**Approach:** [Logistic regression](../05-supervised-learning.md) on vital signs and lab values updated in real-time from the EHR. Alert fires when the patient's EDI score crosses a threshold.

**Key technical decisions (and their failures):**
- **UCSF/Michigan independent validation** (published 2021): the EDI had AUC ≈ 0.74-0.76 on their patient populations — materially lower than Epic's reported 0.83 in the original validation. Difference attributed to population characteristics, implementation variation, and possible overfitting to Epic's training health system.
- **Alert fatigue**: a 2021 Michigan Medicine study found the model generated so many alerts that nurses were ignoring them. The *positive predictive value* (fraction of alerts that led to confirmed sepsis) was low, and clinical staff had no way to calibrate which alerts to prioritise. High sensitivity + low PPV = alarm fatigue + automation bias.
- **Feedback loop**: treating a patient in response to an alert changes the outcome — the patient may not deteriorate. This makes prospective model validation extremely difficult; the outcome being predicted is partially under the control of the model's own alerts.

**Transferable lesson:** A model deployed at scale, firing hundreds of alerts per day at exhausted nurses, will fail operationally even if the AUC is excellent. **Alert fatigue** is the primary failure mode of clinical early-warning systems. Optimise for *actionable* precision at a clinically meaningful threshold, not sensitivity alone.

---

## Cross-cutting lessons

1. **Curated validation ≠ deployment performance.** Always test on data from the deployment context before shipping.
2. **Calibration is a clinical requirement**, not a nice-to-have. Risk scores used by clinicians must be calibrated; probabilities must mean probabilities.
3. **Alert fatigue** kills clinical ML. High-sensitivity models with low PPV harm patients by exhausting staff response capacity.
4. **Fairness and subgroup auditing** are not optional in healthcare — they're regulatory and ethical requirements.
