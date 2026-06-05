# Clinical Risk Modeling Report

## 1. Executive Summary

This report currently summarizes preliminary exploratory data analysis for the UCI Heart Disease processed Cleveland dataset. No machine learning models have been trained yet, and this project should not be interpreted as a diagnostic medical tool.

## 2. Biomedical Question

- Population:
- Prediction target:
- Prediction time point:
- Candidate predictors:
- Intended use:

## 3. Dataset

- Dataset name: UCI Heart Disease, processed Cleveland file
- Source: UCI Machine Learning Repository
- Access date: TODO: document local download date
- Local raw file: `data/raw/processed.cleveland.data`
- Final cohort size in local file: 303 rows
- Original variables: 14 columns
- Exploratory binary target distribution: 164 records with `target_binary = 0`; 139 records with `target_binary = 1`

Refer to `data/README_data.md` for the complete data card.

## 4. Methods

### 4.1 Preprocessing

- Missing data approach:
- Categorical encoding:
- Numeric scaling:
- Excluded variables:
- Leakage prevention checks:

### 4.2 Feature Engineering

- Baseline predictors:
- Derived variables:
- Clinical rationale:

### 4.3 Modeling

- Baseline model:
- Train/test or cross-validation strategy:
- Class imbalance handling:
- Hyperparameters:

### 4.4 Evaluation

Planned metrics:

- Discrimination: ROC AUC and/or average precision
- Calibration: calibration curve and Brier score when appropriate
- Threshold-based performance: sensitivity, specificity, precision, recall, and F1 score

## 5. Preliminary Exploratory Data Analysis

EDA was run locally from `notebooks/01_exploratory_data_analysis.ipynb`. The raw dataset is not tracked in Git.

### 5.1 Data Quality Summary

- Shape: 303 rows and 14 original columns.
- Missing values: `ca` has 4 missing values; `thal` has 2 missing values.
- Duplicate rows: 0.
- Original target distribution: `0` = 164, `1` = 55, `2` = 36, `3` = 35, `4` = 13.
- Exploratory binary target distribution: `0` = 164, `1` = 139.

### 5.2 Main Descriptive Observations

- Age ranges from 29 to 77 years, with a mean of 54.44 years.
- Serum cholesterol ranges from 126 to 564 mg/dl, with a mean of 246.69 mg/dl.
- Maximum heart rate achieved ranges from 71 to 202, with a mean of 149.61.
- In descriptive group summaries, the `target_binary = 1` group has a higher mean age and lower mean maximum heart rate than the `target_binary = 0` group in this dataset.

These are dataset-level descriptive observations only. They should not be interpreted as causal effects, diagnostic rules, or clinical recommendations.

## 6. Biomedical Interpretation

At this stage, interpretation is limited to data quality and descriptive patterns. The processed Cleveland dataset can support educational exploration of clinical risk modeling workflows, but no model has been trained and no individual-level clinical conclusions should be drawn.

## 7. Limitations

- The dataset is historical and may not reflect contemporary clinical practice.
- The processed Cleveland file is a reduced 14-variable version of the original dataset.
- Missingness in `ca` and `thal` must be handled transparently before modeling.
- Several variables are encoded categories and should not be over-interpreted without documentation.
- Generalizability is limited.
- Clinical deployment is outside the scope of this repository.

## 8. Reproducibility

Document the exact commands used to reproduce preprocessing, modeling, evaluation, and figures.

```bash
python -m pip install -r requirements.txt
python -m pytest
python -m ruff check .
```

## 9. Conclusion

Preliminary EDA is complete. The next phase should define preprocessing choices and leakage checks before any baseline modeling is added.
