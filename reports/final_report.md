# Clinical Risk Modeling Report

## 1. Executive Summary

Briefly summarize the biomedical question, dataset, model type, main evaluation results, and key limitations. Do not include claims until supported by real analysis.

## 2. Biomedical Question

- Population:
- Prediction target:
- Prediction time point:
- Candidate predictors:
- Intended use:

## 3. Dataset

- Dataset name:
- Source:
- Access date:
- Inclusion criteria:
- Exclusion criteria:
- Final cohort size:
- Outcome prevalence:

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

## 5. Results

Add results only after running the full analysis on a documented dataset.

| Metric | Estimate | Notes |
| --- | ---: | --- |
| ROC AUC | TBD | Not yet evaluated |
| Average precision | TBD | Not yet evaluated |
| Brier score | TBD | Not yet evaluated |

## 6. Biomedical Interpretation

Interpret the model as a predictive tool within the limits of the dataset. Avoid causal language unless a causal design is explicitly used.

## 7. Limitations

- Data source limitations:
- Missingness and measurement limitations:
- Potential selection bias:
- Potential leakage risks:
- Generalizability:
- Clinical deployment constraints:

## 8. Reproducibility

Document the exact commands used to reproduce preprocessing, modeling, evaluation, and figures.

```bash
python -m pip install -r requirements.txt
pytest
```

## 9. Conclusion

To be completed after real data analysis. State findings cautiously and tie them directly to the evaluation results.
