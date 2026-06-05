# Clinical Risk Modeling Report

## 1. Executive Summary

This report summarizes exploratory data analysis, an interpretable logistic regression baseline, and internal validation for the UCI Heart Disease processed Cleveland dataset. This project should not be interpreted as a diagnostic medical tool or a clinically validated risk model.

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

- Baseline predictors: 13 variables from the processed Cleveland dataset.
- Numeric variables: `age`, `trestbps`, `chol`, `thalach`, and `oldpeak`.
- Categorical variables: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, and `thal`.
- Derived variable: `target_binary`, with original target `0` mapped to `0` and values greater than `0` mapped to `1`.
- Clinical rationale: variables are retained as documented dataset features; no causal interpretation is assigned.

### 4.3 Modeling

- Baseline model: logistic regression in a scikit-learn pipeline.
- Comparison baseline: majority-class `DummyClassifier`.
- Train/test strategy: stratified 80/20 split with `random_state=42`.
- Class imbalance handling: logistic regression uses `class_weight="balanced"`.
- Hyperparameters: simple defaults with `max_iter=1000`, `solver="liblinear"`.

### 4.4 Evaluation

Reported metrics:

- Discrimination: ROC AUC and average precision
- Calibration: calibration curve and Brier score
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

### 7.1 Modeling Objective

The Phase 4 objective is to establish a transparent baseline workflow for binary educational risk modeling. The original `target` column is preserved, and `target_binary` maps original `0` to `0` and original values greater than `0` to `1`.

### 7.2 Train/Test Split

The dataset was split into 242 training rows and 61 held-out test rows using stratified splitting. The held-out test set contains 33 records with `target_binary = 0` and 28 records with `target_binary = 1`.

### 7.3 Baseline Models

- Dummy baseline: majority-class `DummyClassifier`.
- Logistic regression baseline: missing-value imputation, numeric scaling, categorical one-hot encoding, and logistic regression in a single scikit-learn pipeline.

### 7.4 Held-Out Test Metrics

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy majority baseline | 0.541 | 0.000 | 0.000 | 0.000 | 0.500 | 0.459 |
| Logistic regression baseline | 0.869 | 0.812 | 0.929 | 0.867 | 0.966 | 0.963 |

### 7.5 Cautious Interpretation

The logistic regression baseline performs better than the dummy baseline on this single held-out split. This result should be treated as an exploratory benchmark for the repository workflow, not as evidence of medical validity or readiness for clinical use.

### 7.6 Baseline Modeling Limitations

- Results are from one historical dataset.
- No external validation has been performed.
- Calibration estimates remain uncertain because the held-out sample is small.
- Coefficients describe model behavior under this preprocessing setup and should not be interpreted causally.
- The model should not be used for diagnosis, treatment, or individual medical decisions.

### 7.7 Next Steps

- Review the internal validation and calibration findings below.
- Consider external validation before any claims about transportability.
- Keep future models simple until the baseline is fully documented.

## 8. Model Validation and Calibration

### 8.1 Why Validation Is Needed

A single train/test split can give an unstable estimate, particularly for a small dataset. Phase 5 therefore uses stratified cross-validation within the training partition to estimate internal variability while preserving the held-out test set for one final evaluation.

### 8.2 Validation Design

- Held-out design: stratified 80/20 split with 242 training rows and 61 test rows.
- Cross-validation design: five-fold stratified cross-validation on training data only.
- Leakage control: imputation, scaling, one-hot encoding, and logistic regression are refitted within each training fold through the scikit-learn pipeline.
- Final evaluation: a fresh pipeline is fitted on all training rows and evaluated once on the held-out test set.

### 8.3 Cross-Validation Results

| Metric | Training CV Mean | Training CV SD | Held-Out Test |
| --- | ---: | ---: | ---: |
| Accuracy | 0.855 | 0.027 | 0.869 |
| Precision | 0.873 | 0.069 | 0.812 |
| Recall | 0.810 | 0.069 | 0.929 |
| F1-score | 0.837 | 0.029 | 0.867 |
| ROC-AUC | 0.902 | 0.017 | 0.966 |
| Average precision | 0.899 | 0.025 | 0.963 |

The cross-validation means show relatively limited fold-to-fold variation for this training partition, although precision and recall vary more than accuracy, F1, and discrimination metrics. The held-out values fall near or above the cross-validation means, but one favorable test split cannot establish generalizability.

### 8.4 Calibration Analysis

The held-out Brier score is 0.083. The calibration curve is broadly ordered, but some middle-probability bins differ noticeably from their observed positive fractions. With only 61 held-out records, each calibration bin contains few observations, so the curve is too uncertain to support claims that the probabilities are clinically calibrated.

### 8.5 Threshold Analysis

Thresholds from 0.20 to 0.80 were examined descriptively on the held-out test set. As the threshold increased:

- Recall decreased from 1.000 at 0.20 to 0.750 at 0.80.
- Specificity increased from 0.606 at 0.20 to 0.970 at 0.80.
- Precision increased from 0.683 at 0.20 to 0.955 at 0.80.

This illustrates that a probability threshold is not clinically neutral. It changes the balance between false positives and false negatives. Selecting a threshold would require a predefined use case, explicit error costs, prevalence context, and validation in an appropriate external population. No clinical threshold is selected here.

### 8.6 Why This Is Not Clinical Validation

This work is internal validation on a small historical dataset. It does not evaluate transportability across institutions, time periods, populations, measurement systems, or clinical workflows. It also does not assess prospective impact, fairness, decision utility, or patient outcomes.

### 8.7 Next Steps

- Quantify calibration uncertainty with resampling in a later phase.
- Evaluate the unchanged pipeline on an external dataset if a suitable permitted source is identified.
- Define any future threshold analysis from a stated decision context rather than optimizing on the test set.
- Avoid adding model complexity until the validation objective is clear.

## 9. Overall Limitations

- The dataset is historical and may not reflect contemporary clinical practice.
- The processed Cleveland file is a reduced 14-variable version of the original dataset.
- Missingness in `ca` and `thal` must be handled transparently before modeling.
- Several variables are encoded categories and should not be over-interpreted without documentation.
- Generalizability is limited.
- Internal cross-validation does not replace external validation.
- Threshold analysis is descriptive and does not establish decision utility.
- Clinical deployment is outside the scope of this repository.

## 10. Reproducibility

Document the exact commands used to reproduce preprocessing, modeling, evaluation, and figures.

```bash
python -m pip install -r requirements.txt
python -m pytest
python -m ruff check .
```

## 11. Conclusion

EDA, an interpretable baseline, and internal validation are complete. The results support the reproducibility of the workflow within this dataset, but they do not establish clinical validity. Further work should prioritize external validation and calibration uncertainty before considering more complex models.
