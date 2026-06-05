# Model Card: Logistic Regression Baseline

## Model Details

- **Model name:** UCI Heart Disease logistic regression baseline
- **Model type:** Binary logistic regression in a scikit-learn pipeline
- **Project version:** 1.0.0
- **Author:** Jenner Feijoo
- **Status:** Educational portfolio model; not clinically validated

## Dataset

The model uses the processed Cleveland subset of the UCI Heart Disease
dataset: 303 records, 13 predictors, and the original `num` outcome loaded as
`target`.

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart
Disease* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C52P4X>

Raw data are not redistributed by this repository.

## Intended Use

- Demonstrate reproducible preprocessing for mixed clinical tabular data
- Provide an interpretable baseline for educational model evaluation
- Illustrate cross-validation, calibration, and threshold tradeoffs

## Not Intended Use

- Medical diagnosis, screening, triage, treatment, or prognosis
- Individual patient decision-making
- Deployment in a clinical workflow
- Claims of causal effects or clinical utility

## Inputs

Five numeric variables (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) and
eight coded categorical variables (`sex`, `cp`, `fbs`, `restecg`, `exang`,
`slope`, `ca`, `thal`).

Numeric values receive median imputation and standard scaling. Categorical
values receive most-frequent imputation and one-hot encoding. All preprocessing
is fitted inside the model pipeline.

## Output

The pipeline returns a probability and binary class for `target_binary`, where
the original target value `0` maps to `0` and values greater than `0` map to
`1`. The notebooks report threshold `0.50` for baseline evaluation, but no
clinical threshold is endorsed.

## Training and Evaluation Design

- Stratified 80/20 split with `random_state=42`
- 242 training records and 61 held-out test records
- Five-fold stratified cross-validation on training data only
- Final pipeline fitted on the full training partition
- Held-out test set evaluated once for the reported baseline

## Metrics Summary

| Metric | Training CV Mean (SD) | Held-Out Test |
| --- | ---: | ---: |
| Accuracy | 0.855 (0.027) | 0.869 |
| Precision | 0.873 (0.069) | 0.812 |
| Recall | 0.810 (0.069) | 0.929 |
| F1-score | 0.837 (0.029) | 0.867 |
| ROC-AUC | 0.902 (0.017) | 0.966 |
| Average precision | 0.899 (0.025) | 0.963 |

These values describe internal evaluation on a small historical dataset and do
not estimate performance in another population.

## Calibration Summary

The held-out Brier score is `0.083`. The calibration curve is broadly ordered,
with visible uncertainty in middle-probability regions. The 61-record test set
is too small to establish clinical calibration.

## Threshold Note

Across thresholds from `0.20` to `0.80`, increasing the threshold raises
specificity and precision while reducing recall. Threshold selection is not
clinically neutral and would require a prespecified use case, error costs,
prevalence context, and external validation. No operational threshold is
recommended.

## Ethical and Clinical Limitations

- Historical dataset with limited population and collection context
- Binary outcome derived from a historical angiographic label
- Source coding for sex is binary and incomplete for contemporary use
- No external, temporal, prospective, or multi-site validation
- No assessment of subgroup fairness, decision utility, or patient outcomes
- Coefficients represent model behavior and are not causal clinical effects

## Known Failure Modes

- Performance and calibration may change under population or prevalence shift.
- Missingness patterns may differ from the development dataset.
- Unseen categorical values are ignored by the encoder and may reduce validity.
- Measurements collected under different protocols may not be comparable.
- A threshold chosen on this test set may overfit the small evaluation sample.

## Future Validation Needs

- External validation in a separately sourced, permitted cohort
- Calibration uncertainty estimated with resampling
- Prespecified subgroup and fairness analyses with adequate sample sizes
- Decision-curve or utility analysis tied to a clearly defined use case
- Prospective evaluation before any consideration of clinical use
