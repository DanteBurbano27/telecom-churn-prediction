# Telecom Churn Prediction

A portfolio analysis of whether customer behavior, account, and device signals can rank telecom
customers by churn risk. The corrected workflow selects a model with training-only cross-validation
and evaluates it once on an untouched holdout. Its intended use is to design retention experiments;
it does not demonstrate production deployment, causal retention impact, or realized revenue.

**Project status:** the code, synthetic tests, and CI are reproducible publicly. A corrected empirical
rerun is pending access to the course-provided source files. Historical notebook results are retained
below with their validation limitation clearly stated.

## Business Problem

Mass retention campaigns spend incentives on customers who may never churn. A useful model can rank
customers so a team can test offers on a limited high-risk group. Model scores alone do not establish
that an offer prevents churn; that requires a randomized or otherwise causally credible campaign.

## Dataset

The analysis expects two course-provided files:

- `Client.csv`: customer/account/device attributes (historically observed as 100,000 rows, 50 columns)
- `Record.csv`: behavior plus the `churn` target (historically observed as 100,000 rows, 51 columns)

They join one-to-one on `Customer_ID` into 100,000 rows and 100 columns. Raw data are intentionally
not committed. The loader rejects missing, duplicate, and unmatched identifiers rather than silently
changing the analytical population.

## Dataset provenance

The exact release, retrieval URL, and license for these two files are **unresolved**. They were supplied
for a Global Consumer Intelligence course assignment and the original notebook loaded them from the
author's private Google Drive. No repository evidence authorizes redistribution.

The field names and missingness patterns are consistent with data described by the Teradata Center for
CRM at Duke University, but that does not prove that this course extract is identical to a public
release. In particular, a commonly cited Cell2Cell download has 71,047 rows and 58 attributes, while
this project used two 100,000-row tables that merge to 100 columns. The repository therefore does not
present that download as a substitute.

Useful source context, pending confirmation from the course provider:

- [Original research description of the Teradata CRM data](https://www.sciencedirect.com/science/article/abs/pii/S0167923604002040)
- [Common Cell2Cell redistribution with a different shape](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)

Before using the data, obtain `Client.csv` and `Record.csv` from the authorized course/provider channel,
confirm its license and permitted use, and place both files in `data/raw/`. See
[`data/README.md`](data/README.md) for the checklist. Do not download a similarly named dataset and
assume schema equivalence.

## Data shape

Saved output from the original notebook recorded:

| Stage | Rows | Columns |
| --- | ---: | ---: |
| `Client.csv` | 100,000 | 50 |
| `Record.csv` | 100,000 | 51 |
| One-to-one merged table | 100,000 | 100 |

The target contained 50,438 non-churn and 49,562 churn records. That near-50/50 composition must not be
treated as an operational churn prevalence. Literature describing related Teradata data reports
oversampling of churners, so model probabilities, precision, and campaign economics require validation
on a population-representative sample.

## Validation design

The corrected pipeline:

1. validates the two input tables and one-to-one join;
2. creates a stratified 80/20 train/test split with `random_state=7` **before** learned preprocessing;
3. learns median imputation, scaling, and categorical encoding inside each training fold;
4. compares Logistic Regression and Random Forest with five-fold stratified cross-validation on the
   training partition using ROC-AUC;
5. selects by mean cross-validation ROC-AUC; and
6. fits the selected pipeline on all training rows and evaluates it once on the untouched test set.

No trustworthy observation-date field or sampling sequence is documented in this repository. A temporal
split should be preferred only if the provider confirms a suitable time field and prediction horizon.
Until then, the stratified random split is explicit and reproducible, with this limitation documented.

## Preprocessing

- `Customer_ID` is excluded from modeling.
- Numeric missing values use training-fold medians.
- Categorical missing values use the explicit `Unknown` level.
- Rare categorical values are grouped by `OneHotEncoder(min_frequency=5)` and unseen test levels are
  ignored safely.
- Logistic Regression receives standardized numeric features; Random Forest receives unscaled values.

The CLI also records missingness and categorical cardinality so high-missing and high-cardinality fields
remain visible for review. Feature timing remains a limitation until the exact data dictionary is found;
the pipeline cannot prove from column names alone that every feature predates the churn outcome.

## Models evaluated

- Logistic Regression
- Random Forest (200 trees, depth 10, minimum 20 samples per leaf)

No Gradient Boosting or XGBoost implementation is claimed.

## Results

The only current empirical values are **historical saved notebook outputs from the original workflow**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.596 | 0.593 | 0.594 | 0.593 | 0.633 |
| Random Forest | 0.615 | 0.606 | 0.636 | 0.621 | 0.666 |

These are not presented as corrected-pipeline results. The original notebook imputed numeric values
before splitting and chose the better model on the same holdout used for reporting. Both practices can
bias evaluation. The corrected pipeline has not been run on the private source data in this repository,
so it intentionally publishes no replacement metrics.

A ROC-AUC of 0.666 is moderate discrimination, not exceptional predictive accuracy. If a clean rerun
produces a similar value, the defensible use is risk ranking for a controlled retention experiment.

## Risk-decile analysis

Historical output placed 1,470 of 2,000 top-decile customers in the churn class: a 73.5% sample churn
rate and 1.48 lift against the near-balanced sample baseline. The top three deciles contained
40.0% of observed churners.

Those values show ranking within the original holdout; they do not establish population precision,
calibrated churn probability, or campaign impact. The corrected CLI regenerates deciles only after
training-only model selection and labels lift relative to the evaluated sample.

## Business interpretation

The project supports one conservative proposition: model-based ranking may help concentrate a retention
pilot among customers with higher observed risk. Any campaign calculation is explicitly hypothetical and
requires all assumptions to be supplied:

```powershell
telecom-churn --data-dir data/raw --output-dir outputs `
  --target-deciles 1 --contact-cost 10 `
  --retention-success-rate 0.20 --annual-customer-value 300
```

`contact-cost`, `retention-success-rate`, and `annual-customer-value` are scenario assumptions, not
observed outcomes. The output uses holdout churn counts for transparent arithmetic and states that a
randomized campaign is required to estimate incremental retention and realized value.

## Limitations

- Exact dataset provenance and licensing remain unresolved.
- Raw data cannot be publicly retrieved from repository instructions alone.
- The class balance is unlikely to represent deployment prevalence.
- Feature timing and the feasibility of temporal validation require the original data dictionary.
- Historical metrics came from a leakage-prone validation workflow and await corrected rerun.
- Random Forest impurity importance can favor continuous or high-cardinality features and is not causal.
- No production service, deployment, adoption, SLA, cost saving, or revenue gain is claimed.

## Reproduce locally

Python 3.10-3.13 is supported. From a clean clone:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

After authorized copies of `Client.csv` and `Record.csv` are placed in `data/raw/`:

```powershell
.\.venv\Scripts\telecom-churn --data-dir data/raw --output-dir outputs
```

The command prints and writes:

- `outputs/summary.json`: data profile, validation design, CV results, and untouched-test metrics;
- `outputs/cv_results.csv`: mean and standard deviation of training-only CV ROC-AUC; and
- `outputs/risk_deciles.csv`: held-out ranking performance.

Run the public deterministic checks without private data:

```powershell
.\.venv\Scripts\python -m compileall -q src tests
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
.\.venv\Scripts\pytest
```

## Repository structure

```text
.
├── .github/workflows/ci.yml          # public deterministic validation
├── data/README.md                    # provenance status and input contract
├── notebooks/
│   └── 01_telecom_churn_intelligence.ipynb
├── presentation/
│   └── Telecom_Churn_Intelligence.pdf # historical presentation; see limitations above
├── src/telecom_churn/
│   ├── cli.py                        # reproducible entry point
│   ├── data.py                       # input and merge integrity checks
│   └── modeling.py                   # CV, holdout, deciles, scenarios
└── tests/                            # synthetic tests; no private data required
```

## Evidence status

| Status | Evidence |
| --- | --- |
| Implemented | Validated loader, training-fold preprocessing, CV selection, untouched test, deciles, scenario CLI |
| Tested | Deterministic synthetic tests and GitHub Actions; no external dataset or service required |
| Demonstrated | Historical notebook outputs, labeled with their original validation limitation |
| Planned | Corrected run and refreshed presentation after authorized source-data access |
| External dependency | Authorized course/provider access to `Client.csv`, `Record.csv`, and their usage terms |
