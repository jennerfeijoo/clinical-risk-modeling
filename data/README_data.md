# Data Card Template

This file will document the dataset used for clinical risk modeling. Complete it before any analysis is presented.

## Dataset Overview

- Dataset name:
- Source organization or publication:
- URL or accession:
- Access date:
- Version or release:
- License or terms of use:
- Data owner or steward:

## Access and Restrictions

- Public, controlled-access, or restricted:
- Approval requirements:
- Redistribution allowed:
- Sensitive data present:
- De-identification status:
- Local storage location:

Do not commit raw or processed patient-level data unless redistribution is explicitly permitted and privacy risk has been reviewed.

## Clinical Context

- Patient population:
- Care setting:
- Inclusion criteria:
- Exclusion criteria:
- Prediction target:
- Prediction time point:
- Follow-up window:

## Variables

| Variable | Type | Description | Role | Missingness Notes |
| --- | --- | --- | --- | --- |
| `example_variable` | numeric/categorical/date | Replace with real description | predictor/outcome/id/exclude | To be assessed |

## Preprocessing Plan

- Identifier handling:
- Date and time handling:
- Missing data handling:
- Outlier or implausible value checks:
- Categorical encoding:
- Numeric scaling:
- Train/test split strategy:

## Known Limitations

- Population representativeness:
- Measurement bias:
- Missingness mechanisms:
- Outcome misclassification risk:
- Temporal leakage risk:
- External validity:

## Ethical and Privacy Notes

- No sensitive, private, restricted, or re-identifiable data should be committed to this repository.
- Any clinical interpretation must be cautious and limited to the dataset context.
- Predictive associations should not be described as causal effects.
