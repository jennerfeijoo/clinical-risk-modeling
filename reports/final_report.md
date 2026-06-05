# Clinical Risk Modeling Report

## 1. Executive Summary

This report currently summarizes preliminary exploratory data analysis and a simple logistic regression baseline for the UCI Heart Disease processed Cleveland dataset. This project should not be interpreted as a diagnostic medical tool or a clinically validated risk model.

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

- Missing data approach: median imputation for numeric variables and most-frequent imputation for categorical variables, fitted on training data only.
- Categorical encoding: one-hot encoding with unknown categories ignored at test time.
- Numeric scaling: standard scaling for numeric variables, fitted on training data only.
- Excluded variables: original multiclass `target` is preserved for documentation but excluded from model features.
- Leakage prevention checks: train/test split is created before fitting preprocessing steps.

### 4.2 Feature Engineering

- Baseline predictors:
- Derived variables:
- Clinical rationale:

### 4.3 Modeling

- Baseline model: logistic regression in a scikit-learn pipeline.
- Comparison baseline: majority-class `DummyClassifier`.
- Train/test strategy: stratified 80/20 split with `random_state=42`.
- Class imbalance handling: logistic regression uses `class_weight="balanced"`.
- Hyperparameters: simple defaults with `max_iter=1000`, `solver="liblinear"`.

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

Interpretation is limited to data quality, descriptive patterns, and the behavior of a simple baseline model on a held-out split. No individual-level clinical conclusions should be drawn.

## 7. Baseline Modeling

### 6.1 Modeling Objective

The Phase 4 objective is to establish a transparent baseline workflow for binary educational risk modeling. The original `target` column is preserved, and `target_binary` maps original `0` to `0` and original values greater than `0` to `1`.

### 6.2 Train/Test Split

The dataset was split into 242 training rows and 61 held-out test rows using stratified splitting. The held-out test set contains 33 records with `target_binary = 0` and 28 records with `target_binary = 1`.

### 6.3 Baseline Models

- Dummy baseline: majority-class `DummyClassifier`.
- Logistic regression baseline: missing-value imputation, numeric scaling, categorical one-hot encoding, and logistic regression in a single scikit-learn pipeline.

### 6.4 Held-Out Test Metrics

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy majority baseline | 0.541 | 0.000 | 0.000 | 0.000 | 0.500 | 0.459 |
| Logistic regression baseline | 0.869 | 0.812 | 0.929 | 0.867 | 0.966 | 0.963 |

### 6.5 Cautious Interpretation

The logistic regression baseline performs better than the dummy baseline on this single held-out split. This result should be treated as an exploratory benchmark for the repository workflow, not as evidence of medical validity or readiness for clinical use.

### 6.6 Baseline Modeling Limitations

- Results are from one historical dataset and one train/test split.
- No external validation has been performed.
- No calibration analysis is included yet.
- Coefficients describe model behavior under this preprocessing setup and should not be interpreted causally.
- The model should not be used for diagnosis, treatment, or individual medical decisions.

### 6.7 Next Steps

- Add cross-validation to assess split sensitivity.
- Add calibration assessment before considering any risk-score interpretation.
- Review preprocessing decisions for missing categorical variables.
- Keep future models simple until the baseline is fully documented.

## 8. Overall Limitations

- The dataset is historical and may not reflect contemporary clinical practice.
- The processed Cleveland file is a reduced 14-variable version of the original dataset.
- Missingness in `ca` and `thal` must be handled transparently before modeling.
- Several variables are encoded categories and should not be over-interpreted without documentation.
- Generalizability is limited.
- Clinical deployment is outside the scope of this repository.

## 9. Reproducibility

Document the exact commands used to reproduce preprocessing, modeling, evaluation, and figures.

```bash
python -m pip install -r requirements.txt
python -m pytest
python -m ruff check .
```

## 10. Conclusion

Preliminary EDA and an interpretable baseline modeling workflow are complete. Further work should focus on validation, calibration, and careful documentation before considering any more complex modeling.
