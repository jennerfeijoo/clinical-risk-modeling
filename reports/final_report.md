# Interpretable Clinical Risk Modeling with the UCI Heart Disease Dataset

## Abstract

This project demonstrates a reproducible and interpretable clinical risk-modeling workflow using the UCI Heart Disease processed Cleveland dataset. The analysis includes data documentation, exploratory analysis, a logistic regression baseline, training-set cross-validation, held-out evaluation, calibration assessment, and threshold analysis. All preprocessing is fitted within scikit-learn pipelines to reduce leakage risk. The held-out results are encouraging for this particular split, but the dataset is small, historical, and unsuitable for establishing clinical validity. The project is educational and must not be used for diagnosis or individual medical decisions.

## 1. Background

Clinical prediction modeling requires methodological transparency as well as predictive performance. Important considerations include outcome definition, missing-data handling, separation of model development and evaluation data, probability calibration, threshold consequences, and external validity.

This project asks:

> Can the documented variables in the processed Cleveland dataset support a reproducible, interpretable binary prediction workflow for educational analysis?

The objective is to demonstrate responsible analytical practice, not to create a medical device or clinically deployable risk score.

## 2. Dataset

The analysis uses the processed Cleveland subset of the UCI Heart Disease dataset.

- Source: UCI Machine Learning Repository
- Local file: `data/raw/processed.cleveland.data`
- Rows: 303
- Original variables: 14
- Raw data status: stored locally and excluded from Git
- Local access date: June 5, 2026

The original outcome variable, `num`, is loaded as `target` and uses values from `0` to `4`. For binary analysis, `target_binary` retains `0` as `0` and maps values greater than `0` to `1`. This transformation follows a common analytical framing of the dataset but does not constitute a diagnosis.

The complete variable dictionary, access instructions, license, and data limitations are documented in [README_data.md](../data/README_data.md).

The intended use, evaluation scope, and known failure modes of the fitted
baseline are summarized separately in the [model card](model_card.md).

## 3. Methods

### 3.1 Data Preparation

The raw comma-separated file is loaded with 14 documented column names. The `?` marker is treated as missing, and analysis columns are converted to numeric values where appropriate. The original target is preserved alongside the derived binary target.

### 3.2 Feature Definition

Five variables are treated as continuous numeric features:

- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`

Eight integer-coded variables are treated as categorical features:

- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`

Treating coded categories as categorical avoids assuming that their numeric codes have a linear relationship with the outcome.

### 3.3 Preprocessing

Preprocessing is contained within a scikit-learn `Pipeline` and `ColumnTransformer`:

- Numeric missing values: median imputation
- Numeric scaling: standardization
- Categorical missing values: most-frequent imputation
- Categorical representation: one-hot encoding with unknown categories ignored

The data are divided using a stratified 80/20 train/test split with `random_state=42`. Preprocessing is fitted only on training data, including within each cross-validation fold.

### 3.4 Models

Two simple models are evaluated:

1. A majority-class `DummyClassifier`
2. Logistic regression with balanced class weights

Logistic regression is retained as the primary model because it provides a transparent baseline and supports inspection of fitted coefficients. Coefficients are interpreted as model parameters under the chosen preprocessing scheme, not as causal effects.

### 3.5 Evaluation Design

Five-fold stratified cross-validation is performed only on the 242-record training partition. A fresh pipeline is then fitted on the full training set and evaluated once on the 61-record held-out test set.

Reported metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average precision
- Confusion matrix
- Brier score
- Calibration curve
- Threshold-dependent precision, recall, specificity, and F1-score

## 4. Exploratory Analysis

### 4.1 Data Quality

- Missing values: `ca` has 4 missing values and `thal` has 2.
- Duplicate rows: 0.
- Original target counts: `0` = 164, `1` = 55, `2` = 36, `3` = 35, `4` = 13.
- Binary target counts: `0` = 164 and `1` = 139.

### 4.2 Descriptive Summary

- Mean age: 54.44 years; range: 29 to 77.
- Mean serum cholesterol: 246.69 mg/dl; range: 126 to 564.
- Mean maximum heart rate achieved: 149.61; range: 71 to 202.

The group coded `target_binary = 1` has a higher mean age and lower mean maximum heart rate than the group coded `0` in this dataset. These are descriptive observations only and do not imply causation or clinical decision rules.

Selected exploratory figures:

- [Age distribution](figures/age_distribution.png)
- [Age by binary target](figures/age_by_target_binary.png)
- [Missing-value summary](figures/missing_values.png)

## 5. Baseline Modeling

The held-out test set contains 33 records coded `0` and 28 coded `1`.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy majority baseline | 0.541 | 0.000 | 0.000 | 0.000 | 0.500 | 0.459 |
| Logistic regression baseline | 0.869 | 0.812 | 0.929 | 0.867 | 0.966 | 0.963 |

The logistic regression pipeline performs better than the majority-class baseline on this held-out split. The result demonstrates that the workflow extracts predictive signal in this dataset; it does not establish medical validity or expected performance in another population.

Selected baseline figures:

- [Held-out confusion matrix](figures/baseline_confusion_matrix.png)
- [ROC curve](figures/baseline_roc_curve.png)
- [Precision-recall curve](figures/baseline_precision_recall_curve.png)
- [Largest fitted coefficients](figures/baseline_logistic_coefficients.png)

## 6. Model Validation and Calibration

### 6.1 Cross-Validation

| Metric | Training CV Mean | Training CV SD | Held-Out Test |
| --- | ---: | ---: | ---: |
| Accuracy | 0.855 | 0.027 | 0.869 |
| Precision | 0.873 | 0.069 | 0.812 |
| Recall | 0.810 | 0.069 | 0.929 |
| F1-score | 0.837 | 0.029 | 0.867 |
| ROC-AUC | 0.902 | 0.017 | 0.966 |
| Average precision | 0.899 | 0.025 | 0.963 |

Fold-to-fold variation is modest for accuracy, F1-score, ROC-AUC, and average precision, while precision and recall vary more. Held-out results are near or above the cross-validation means, but one favorable test partition cannot establish generalizability.

[Cross-validation metric summary](figures/validation_cross_validation_metrics.png)

### 6.2 Calibration

The held-out Brier score is `0.083`. Predicted probabilities are broadly ordered with observed outcome frequency, but middle-probability bins show visible departures from the perfect-calibration line. Because the held-out sample contains only 61 records, the calibration curve is imprecise and cannot support a claim of clinical calibration.

[Held-out calibration curve](figures/validation_calibration_curve.png)

## 7. Threshold Analysis

Thresholds from `0.20` to `0.80` were examined descriptively on the held-out test set.

- Recall decreases from `1.000` at threshold `0.20` to `0.750` at `0.80`.
- Specificity increases from `0.606` at threshold `0.20` to `0.970` at `0.80`.
- Precision increases from `0.683` at threshold `0.20` to `0.955` at `0.80`.

Changing a threshold alters the balance between false positives and false negatives. A threshold is therefore not clinically neutral. Selection would require a prespecified use case, explicit error costs, prevalence context, and validation in a relevant external population. No clinical threshold is selected in this project.

- [Threshold metric comparison](figures/validation_threshold_metrics.png)
- [Confusion matrix at threshold 0.50](figures/validation_test_confusion_matrix_threshold_050.png)

## 8. Ethical and Clinical Scope

- The project is educational and retrospective.
- The outcome is a historical dataset label.
- Predictive associations are not causal effects.
- Model coefficients are not clinical risk factors established by this analysis.
- Results must not be used for diagnosis, treatment, or individual decision-making.
- Internal validation does not demonstrate transportability, safety, fairness, or clinical benefit.

## 9. Limitations

- The dataset is small, historical, and derived from a limited setting.
- The processed file contains only 14 of the original database variables.
- Some variables are integer-coded categories with limited contextual detail.
- Missing values occur in `ca` and `thal`.
- Evaluation uses one held-out partition and no external cohort.
- Calibration estimates are sensitive to the small test sample and binning strategy.
- Threshold comparisons are descriptive and do not establish decision utility.
- The project does not assess temporal drift, subgroup fairness, prospective impact, or patient outcomes.

## 10. Conclusion

The project provides a reproducible example of leakage-aware preprocessing, interpretable baseline modeling, internal validation, and cautious probability assessment for clinical tabular data. The logistic regression pipeline shows predictive signal within the processed Cleveland dataset, with reasonably stable training-set cross-validation estimates. These findings are limited to this educational analysis and do not establish clinical validity.

## 11. Future Work

- Evaluate the unchanged pipeline on a suitable external dataset.
- Quantify uncertainty in calibration estimates using resampling.
- Define threshold analyses from a prespecified decision context.
- Assess subgroup performance only when sample size and variable definitions support responsible interpretation.
- Retain simple models until a clear validation objective justifies additional complexity.

## 12. Reproducibility

Install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Place the official processed Cleveland file at:

```text
data/raw/processed.cleveland.data
```

Run notebooks `01`, `02`, and `03` in order, then run:

```bash
python -m pytest
python -m ruff check .
```

## 13. References

1. Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>
2. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
