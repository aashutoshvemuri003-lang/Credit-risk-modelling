# Credit Risk Modelling — Model Comparison & WoE Scorecard

An end-to-end credit risk pipeline built on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle dataset (150,000 consumer borrowers), comparing eight modelling approaches and culminating in a production-style, industry-standard Weight-of-Evidence (WoE) credit scorecard.

## Problem

Predict the probability that a borrower experiences serious delinquency (90+ days past due) within two years, given 10 borrower-level features (credit utilization, delinquency history, income, debt ratio, etc.). Base event rate: 6.7%.

## Approach

| Stage | Method |
|---|---|
| Baselines | Logistic Regression, LDA, QDA, KNN |
| Non-linearity | Spline Logistic Regression (knot selection via 10-fold CV + 1-SE rule) |
| Ensembles | Random Forest, XGBoost (grid-searched, early stopping) |
| Production candidate | WoE binning → Information Value screening → WoE Logistic Regression → points-based scorecard |

## Results

| Model | Test AUC | Gini |
|---|---|---|
| XGBoost (tuned) | 0.8687 | 0.7374 |
| Random Forest (tuned) | 0.8677 | 0.7354 |
| **WoE Scorecard (recommended)** | 0.8577 | 0.7154 |
| Spline LR (10 knots) | 0.8254 | 0.6508 |
| KNN (k=220) | 0.8056 | 0.6112 |
| QDA | 0.7955 | 0.5909 |
| Logistic Regression | 0.7123 | 0.4246 |
| LDA | 0.7065 | 0.4129 |

**Recommendation:** the WoE scorecard, despite marginally lower AUC than the ensembles, is proposed as the production model. It is fully monotonic and interpretable at the bin level, natively supports adverse-action reason codes required under fair-lending regulation (e.g. ECOA), and its stability was confirmed via Population Stability Index (0.0002, well under the 0.1 "no shift" threshold).

## Key methodological findings

- **Non-linearity matters:** moving from plain logistic regression to a spline formulation raised test AUC by 0.11 — the single largest jump in the comparison — confirming non-linear risk relationships in the data.
- **1-SE rule for knot selection:** rather than the CV-optimal 25 knots, a 10-knot spline was selected as the simplest model within one standard error of peak CV AUC, trading a negligible 0.005 AUC for a substantially more stable model.
- **`scorecardpy.iv()` pitfall (verified empirically):** the package's default `iv()` function computes Information Value by grouping on *every unique raw value* of a variable rather than using coarse, pre-computed bins. For continuous variables this severely inflates IV — confirmed via a synthetic-data test where a variable with *zero true relationship* to the target produced an IV of 0.14 under this method vs. 0.008 under correct coarse binning. All reported IVs in this project were recomputed from `bins[v]['bin_iv'].sum()` to correct for this.
- **Missing-value handling:** for the scorecard, missing values (e.g. ~20% of `MonthlyIncome`) were kept as their own WoE bin rather than imputed, since missingness can itself carry risk signal — though in this dataset, once IV was corrected, `MonthlyIncome`'s overall predictive power (~0.07) turned out to be weak regardless.

## Repo structure

```
├── Credit_Risk_Scorecard.ipynb   # full analysis notebook
├── Credit_Risk_Scorecard_Report.pdf  # polished write-up (exec summary, methodology, results)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Download `cs-training.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in a `data/` folder before running the notebook (dataset is not redistributed in this repo per Kaggle's terms).

## Tools

Python · pandas · scikit-learn · XGBoost · scorecardpy · statsmodels · matplotlib/seaborn
