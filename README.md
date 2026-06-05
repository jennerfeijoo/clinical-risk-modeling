# Clinical Risk Modeling

This repository is a starter project for a Biomedical Data Science for Personalized Medicine portfolio. It is designed for a future clinical risk modeling dataset, with emphasis on reproducible preprocessing, interpretable baseline models, cautious evaluation, and transparent reporting.

No dataset is included yet. This repository must not contain sensitive, private, restricted, or re-identifiable patient data.

## Biomedical Question

Primary question to define after dataset selection:

> Can routinely available clinical variables be used to estimate the risk of a clinically meaningful outcome for a defined patient population?

The final question should specify:

- Patient population and inclusion criteria
- Prediction target and prediction time point
- Candidate predictors available before the prediction time point
- Intended analytical use, such as retrospective risk stratification or baseline modeling

## Planned Workflow

1. Document the dataset source, access date, license, variables, and restrictions in `data/README_data.md`.
2. Place raw data locally under `data/raw/` only if it is public, permitted, and non-sensitive.
3. Create cleaned analysis files under `data/processed/`.
4. Develop readable exploratory analysis in `notebooks/`.
5. Reuse modular code from `src/` for preprocessing, feature construction, modeling, and evaluation.
6. Summarize methods, results, limitations, and interpretation in `reports/final_report.md`.

## Repository Structure

```text
clinical-risk-modeling/
|-- data/
|   |-- raw/                 # Local raw data, not committed
|   |-- processed/           # Local processed data, not committed
|   `-- README_data.md       # Data card template
|-- notebooks/               # Sequential exploratory notebooks
|-- reports/
|   |-- figures/             # Generated figures
|   `-- final_report.md      # Structured report template
|-- src/
|   |-- data_processing.py   # Data loading and cleaning helpers
|   |-- features.py          # Feature engineering helpers
|   |-- modeling.py          # Baseline modeling helpers
|   `-- evaluation.py        # Evaluation metrics and summaries
|-- tests/
|   `-- test_basic.py        # Minimal starter tests
`-- requirements.txt
```

## Reproducibility

Create an environment and install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest
```

Run linting:

```bash
python -m ruff check .
```

## Modeling Principles

- Start with simple, interpretable baselines.
- Separate training and evaluation data before fitting preprocessing steps.
- Avoid leakage from future information, outcome-derived variables, or post-baseline measurements.
- Report discrimination and calibration when appropriate.
- Interpret results as predictive associations, not causal effects.
- Document missingness, cohort construction, exclusions, and limitations.

## Current Status

Project skeleton only. No clinical dataset, results, or claims are included.
