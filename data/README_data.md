# Data Card: UCI Heart Disease Dataset

## Dataset Overview

- Dataset name: Heart Disease
- Source: UCI Machine Learning Repository
- Dataset page: <https://archive.ics.uci.edu/dataset/45/heart+disease>
- DOI: <https://doi.org/10.24432/C52P4X>
- Authors: Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano
- Official citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>
- License: CC BY 4.0, according to the UCI dataset page
- Local access date: June 5, 2026
- Expected raw file: `data/raw/processed.cleveland.data`

## Intended Use in This Repository

This dataset is used for educational portfolio work in Biomedical Data Science for Personalized Medicine. The repository includes exploratory analysis, an interpretable logistic regression baseline, internal cross-validation, calibration assessment, and threshold analysis.

## Not Intended Use

This repository is not a diagnostic medical tool, clinical decision support system, or deployment-ready risk calculator. The analysis should not be used to make individual medical decisions or infer causal effects.

## Access and Placement

The raw dataset is not committed to this repository. To reproduce the notebooks:

1. Go to the official UCI dataset page: <https://archive.ics.uci.edu/dataset/45/heart+disease>.
2. Download the dataset files from UCI.
3. Place the processed Cleveland file at:

```text
data/raw/processed.cleveland.data
```

Do not add restricted, private, sensitive, or re-identifiable patient data to this repository.

## Privacy and Ethics Note

The UCI Heart Disease dataset is a public educational dataset, but it originates from clinical data. Analysis must remain cautious, avoid overstating clinical meaning, and acknowledge that public availability does not remove the need for responsible use. No private patient-level data should be added to this repository.

## Target Variable

The commonly used processed Cleveland file contains a final variable originally named `num`. In this repository it is loaded as `target`.

- Original coding: `0`, `1`, `2`, `3`, `4`
- Educational binary coding used in this repository: `0` remains `0`; values greater than `0` are mapped to `1`

In the original dataset framing, values greater than `0` indicate presence of angiographic heart disease. This should be interpreted only as a dataset label definition, not as a medical claim or diagnostic statement.

## Variable Dictionary

| Variable | Type | Description | Role |
| --- | --- | --- | --- |
| `age` | numeric | Age in years | Predictor |
| `sex` | categorical-coded | Sex code from the source documentation | Predictor |
| `cp` | categorical-coded | Chest pain type code | Predictor |
| `trestbps` | numeric | Resting blood pressure on admission to the hospital, in mm Hg | Predictor |
| `chol` | numeric | Serum cholesterol in mg/dl | Predictor |
| `fbs` | binary-coded | Indicator for fasting blood sugar greater than 120 mg/dl | Predictor |
| `restecg` | categorical-coded | Resting electrocardiographic result code | Predictor |
| `thalach` | numeric | Maximum heart rate achieved | Predictor |
| `exang` | binary-coded | Exercise-induced angina indicator | Predictor |
| `oldpeak` | numeric | ST depression induced by exercise relative to rest | Predictor |
| `slope` | categorical-coded | Peak exercise ST-segment slope code | Predictor |
| `ca` | count/categorical-coded | Number of major vessels colored by fluoroscopy | Predictor |
| `thal` | categorical-coded | Historical thal test result code | Predictor |
| `target` / `num` | categorical-coded | Original angiographic disease label, loaded as `target` in this repository | Outcome label |

## Coded Variable Definitions

The code meanings below are reproduced from the UCI dataset documentation.
They reflect historical dataset terminology and should not be overinterpreted
as contemporary clinical definitions.

| Variable | Documented codes |
| --- | --- |
| `sex` | `0` = female; `1` = male |
| `cp` | `1` = typical angina; `2` = atypical angina; `3` = non-anginal pain; `4` = asymptomatic |
| `fbs` | `0` = fasting blood sugar not above 120 mg/dl; `1` = fasting blood sugar above 120 mg/dl |
| `restecg` | `0` = normal; `1` = ST-T wave abnormality; `2` = probable or definite left ventricular hypertrophy by Estes' criteria |
| `exang` | `0` = no exercise-induced angina; `1` = exercise-induced angina |
| `slope` | `1` = upsloping; `2` = flat; `3` = downsloping |
| `ca` | Integer count from `0` to `3` major vessels colored by fluoroscopy; missing values occur in the processed Cleveland file |
| `thal` | `3` = normal; `6` = fixed defect; `7` = reversible defect; missing values occur in the processed Cleveland file |

## Missing Data

The processed Cleveland file uses `?` to represent missing values. The loading and cleaning code converts these markers to pandas missing values before analysis.

## Known Limitations

- The dataset is historical and may not reflect contemporary clinical practice or current population distributions.
- The commonly used 14-variable processed version is a reduced representation of the original database.
- Several variables are categorical codes that require careful documentation before interpretation.
- The sex variable uses a binary coding scheme and terminology from the source documentation; it does not represent the full range of sex or gender information relevant to contemporary analysis.
- Missingness is present in some variables and should be summarized before analysis.
- The dataset is not sufficient on its own to support clinical deployment claims.
- Any observed associations are predictive or descriptive only, not causal.
