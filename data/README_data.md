# Data Card: UCI Heart Disease Dataset

## Dataset Overview

- Dataset name: Heart Disease
- Source: UCI Machine Learning Repository
- Dataset page: <https://archive.ics.uci.edu/dataset/45/heart+disease>
- DOI: <https://doi.org/10.24432/C52P4X>
- Authors: Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano
- Official citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C52P4X>
- License: CC BY 4.0, according to the UCI dataset page
- Access date: TODO: add date when the raw file is downloaded
- Expected raw file: `data/raw/processed.cleveland.data`

## Intended Use in This Repository

This dataset is used for educational portfolio work in Biomedical Data Science for Personalized Medicine. The current phase is limited to data documentation and exploratory data analysis. Future phases may add simple, interpretable baseline modeling after the data source, preprocessing decisions, and limitations are documented.

## Not Intended Use

This repository is not a diagnostic medical tool, clinical decision support system, or deployment-ready risk calculator. The analysis should not be used to make individual medical decisions or infer causal effects.

## Access and Placement

The raw dataset is not committed to this repository. To run the exploratory notebook:

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
- Educational binary coding for future modeling: `0` remains `0`; values greater than `0` are mapped to `1`

In the original dataset framing, values greater than `0` indicate presence of angiographic heart disease. This should be interpreted only as a dataset label definition, not as a medical claim or diagnostic statement.

## Variable Dictionary

| Variable | Type | Description | Role |
| --- | --- | --- | --- |
| `age` | numeric | Age in years | Predictor |
| `sex` | categorical-coded | Sex coded as `1` = male, `0` = female in the dataset documentation | Predictor |
| `cp` | categorical-coded | Chest pain type | Predictor |
| `trestbps` | numeric | Resting blood pressure on admission to the hospital, in mm Hg | Predictor |
| `chol` | numeric | Serum cholesterol in mg/dl | Predictor |
| `fbs` | binary-coded | Fasting blood sugar greater than 120 mg/dl, coded as `1` = true, `0` = false | Predictor |
| `restecg` | categorical-coded | Resting electrocardiographic results | Predictor |
| `thalach` | numeric | Maximum heart rate achieved | Predictor |
| `exang` | binary-coded | Exercise-induced angina, coded as `1` = yes, `0` = no | Predictor |
| `oldpeak` | numeric | ST depression induced by exercise relative to rest | Predictor |
| `slope` | categorical-coded | Slope of the peak exercise ST segment | Predictor |
| `ca` | numeric/categorical-coded | Number of major vessels colored by fluoroscopy | Predictor |
| `thal` | categorical-coded | Thalassemia-related categorical code | Predictor |
| `target` / `num` | categorical-coded | Original angiographic disease label, loaded as `target` in this repository | Outcome label |

## Missing Data

The processed Cleveland file uses `?` to represent missing values. The loading and cleaning code converts these markers to pandas missing values before analysis.

## Known Limitations

- The dataset is historical and may not reflect contemporary clinical practice or current population distributions.
- The commonly used 14-variable processed version is a reduced representation of the original database.
- Several variables are categorical codes that require careful documentation before interpretation.
- Missingness is present in some variables and should be summarized before analysis.
- The dataset is not sufficient on its own to support clinical deployment claims.
- Any observed associations are predictive or descriptive only, not causal.
