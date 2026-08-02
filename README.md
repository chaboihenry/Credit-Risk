
# Credit Risk Modeling & Model Validation (Lending Club)

This project does two things: it builds a machine learning model that predicts which loan borrowers will default, and it then **audits that entire pipeline the way a bank's model validation team would** — finding and fixing serious flaws, and rejecting one model outright.

The original version of this project reported impressive numbers. The validation work showed several of them were misleading. This README documents both the corrected results and the problems found, because in credit risk, knowing *why* a number can't be trusted matters as much as the number.

### The Two Models

1. **Gatekeeper Model — REJECTED.** An XGBoost classifier meant to replicate Lending Club's accept/reject decision across ~29.9M applications. Validation found it unfit for use (details below). It is kept in the repo as a documented case study, not a working model.
2. **Default Probability Model — CORRECTED & KEPT.** An XGBoost model predicting "Charged Off" vs "Fully Paid" on **1.35M completed loans** (2007–2018), rebuilt after validation and evaluated honestly.

### What the Validation Found & Fixed (Default Model)

| Issue               | Before                                                                                                                                             | After                                                                                        |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Test split          | Random & stratified — future loans leaked into training, and the test set was forced to match historical default rates                            | **Out-of-time split**: trained on 2007–2016, tested on 2017–2018                     |
| Hidden risk drift   | Masked by stratification                                                                                                                           | Test-period default rate is 21.3% vs 19.7% in training — later loans genuinely default more |
| Probability quality | Average predicted default chance:**69%** (actual: 21%) — caused by stacking two class-imbalance corrections                                 | Average predicted:**22.6%** vs actual 21.3%                                            |
| Headline metric     | 0.7259 — cross-validation on an artificially balanced training set; test-set AUC was never computed                                               | **0.7181 AUC on true out-of-time test data**                                           |
| Decision threshold  | 0.89 (an artifact of inflated probabilities)                                                                                                       | 0.51, chosen by profit simulation                                                            |
| Profit claim        | "$198M" — gross profit, wrongly credited entirely to the model |**+$23.4M vs. approving everyone** ($137.7M vs a $114.3M no-model baseline) |                                                                                              |

### Why the Gatekeeper Was Rejected

1. **It can never be checked.** It predicts whether Lending Club *approved* an application — but rejected applications have no outcome, so there is no way to know if any rejection was correct.
2. **Its scores were measured on a fake population.** The test set was built 50/50 accepted/rejected; the real acceptance rate is ~7.6%.
3. **The data contained an answer key.** A `policy_code` column perfectly separated the two classes (caught and removed before training).
4. **A hidden leak was quantified.** Whether a credit score was even *recorded* predicts acceptance with 0.834 AUC by itself — 96% of the full model's power from one data-entry artifact. See `Gatekeeper_Rejected_Model_Case_Study.ipynb`.

### Honest Limitations

The default model is only valid for applicants resembling historically *approved* borrowers (the classic "reject inference" problem — rejected applicants have no repayment data). The profit simulation assumes full-term simple interest, a 10% recovery on defaults, and 2026 funding costs, with no prepayment modeling.

### Project Structure

* `Default_Probability_Model.ipynb` — data cleaning, feature engineering, out-of-time validation, calibrated model, profit optimization.
* `Gatekeeper_Rejected_Model_Case_Study.ipynb` — full write-up of the rejected model, with evidence for each finding.
* `Models/` — versioned artifacts: `default_risk_model_v1` (original, kept for comparison) and `default_risk_model_v2` (corrected).

**Stack:** Python, pandas, XGBoost, scikit-learn, Docker (CPU-only, fully reproducible).
