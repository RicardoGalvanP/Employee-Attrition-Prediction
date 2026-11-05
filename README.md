# Employee Attrition Prediction (IBM HR Analytics)

Binary classification of employee attrition with scikit-learn: leakage-free pipelines, regularized logistic regression vs. KNN, repeated stratified cross-validation, decision-threshold analysis, and an ablation of legally sensitive attributes (age, gender, marital status) with a per-group performance audit.

Course work: M.Sc. in Applied Artificial Intelligence, Tecnológico de Monterrey (AI & Machine Learning course).
Tools: Python, pandas, scikit-learn, seaborn, matplotlib.
AI assistance: [NAME THE LLM(S) YOU USED]. Used for hyperparameter-search ideas and for reviewing variable transformations; outputs were checked against the scikit-learn documentation and my class notes.

## Problem

Losing an employee costs recruiting, training and lost productivity. The goal is to flag employees at risk of leaving so HR can open a retention conversation earlier. Dataset: IBM HR Analytics Employee Attrition (1,470 rows, synthetic, from Kaggle). The positive class (Attrition = Yes) is 16%, so a model that always predicts "No" already scores 83.9% accuracy. That is the baseline every model has to beat.

## Approach

1. Dropped 4 non-informative columns (EmployeeCount, StandardHours, Over18 are constant; EmployeeNumber is an ID).
2. Stratified 70/15/15 train/validation/test split. The analysis used to choose transformations (distributions, skewness, outliers, correlations, VIF) used the training set only.
3. All preprocessing lives inside a scikit-learn Pipeline, so it is refit within every cross-validation fold (no leakage). A ColumnTransformer handles each variable type:
   - numeric (14): median imputation, Yeo-Johnson (many are right-skewed), standardization
   - ordinal (9): mode imputation, standardization
   - binary (2): ordinal encoding
   - nominal (5): one-hot encoding
4. Hyperparameter search (GridSearchCV, 5-fold x 3 repeats) for logistic regression (no penalty, L1, L2, ElasticNet) and KNN; models re-evaluated on different CV splits to reduce selection optimism.
5. Grid search for the final ElasticNet model, a single evaluation on the untouched test set, then a recall-oriented extension (threshold tuning, class weights, attribute ablation, group audit).

## Results

Cross-validated accuracy (train+validation, 1,249 rows; majority baseline = 83.9%):

| Model | Best hyperparameters | Accuracy (mean ± std) |
|---|---|---|
| Logistic regression | no penalty | 0.885 ± 0.018 |
| Lasso (L1) | C = 0.3 | 0.889 ± 0.017 |
| Ridge (L2) | C = 0.3 | 0.889 ± 0.020 |
| ElasticNet | C = 1, l1_ratio = 0.3 | 0.889 ± 0.020 |
| KNN | k = 5 | 0.849 ± 0.009 |

The four logistic variants are indistinguishable (the spread, 0.0035, is far smaller than one standard deviation); regularization did not help. KNN is the weakest and only 1.0 point above the baseline. The final ElasticNet shows no overfitting (train 90.7% vs. validation 88.9%).

### Why accuracy is not the headline

On the held-out test set (221 employees, 36 leavers) the default model reaches 0.860 accuracy against a 0.837 baseline: +2.3 points, with a 95% interval (0.81-0.90) that includes the baseline. What matters is how many leavers are caught.

| Model on test set | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Employees flagged |
|---|---|---|---|---|---|---|---|
| Default threshold 0.5 | 0.860 | 0.61 | 0.39 | 0.47 | 0.727 | 0.504 | 23 of 221 |
| Tuned threshold (F2, from CV) | 0.688 | 0.29 | 0.64 | 0.40 | 0.727 | 0.504 | 79 of 221 |
| Without age, gender, marital status + tuned threshold | 0.710 | 0.31 | 0.64 | 0.42 | 0.719 | 0.493 | 74 of 221 |

- With the default threshold the model catches 14 of 36 leavers (recall 0.39, 95% CI 0.25-0.55).
- Lowering the threshold catches about 2 of 3 leavers, but it means flagging a third of the workforce, with roughly 70% false alarms. It is the same model at a different operating point (identical AUC), a trade-off rather than an improvement. Whether it pays off depends on the cost of a retention intervention relative to the cost of losing an employee.
- `class_weight='balanced'` lands at a similar operating point with a slightly lower ROC-AUC (0.715).
- Out-of-fold ROC-AUC on train+validation is 0.845 versus 0.727 on the test set. With only 36 leavers in the test set the AUC standard error is about 0.05, so part of the gap is noise and part is cross-validation optimism; the cross-validated figure (201 leavers) is the more stable estimate.

### Sensitive attributes

Removing age, gender and marital status changes ROC-AUC from 0.845 to 0.842 in cross-validation (0.727 to 0.719 on test): no meaningful performance cost, so there is no performance reason to keep them. Removing them does not remove disparities, though. Without those attributes, recall for leavers is still 0.60 for employees aged 40+ versus 0.83 for those under 30, and 0.64 for married versus 0.81 for single employees (out-of-fold; the 40+ group has only 47 leavers, so the gap is uncertain). Correlated variables such as total working years, tenure and job level are plausible proxies.

## What the model associates with leaving

Higher attrition risk: working overtime (strongest; 32% attrition rate with overtime vs. 10% without in the training data), frequent business travel, single marital status, Laboratory Technician role.
Lower attrition risk: higher monthly income, Research Director role, higher environment satisfaction, more years in the current role.

These are associations in a synthetic dataset, not causes. Income and job level are almost collinear (r = 0.95, VIF about 11), so individual coefficients for them should not be read in isolation.

## Responsible use

Alerts should go to HR only, as a prompt for a conversation, never as an automatic decision about a person or as a justification for a negative action. Age, gender and marital status should be excluded, and group-level performance should be audited before any deployment.

## Limitations and next steps

- Only 36 leavers in the test set: test-set differences of a few cases are inconclusive. Repeated or nested cross-validation would give tighter estimates.
- Try tree-based models and probability calibration; tie the threshold to a real cost estimate of intervention vs. replacement.
- Investigate and mitigate the recall gap across age and marital-status groups (proxy variables).

## Reproducing

Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
Open the notebook in Jupyter or Colab. Random seeds are fixed (split 17, models 1, CV 7 and 11).
