"""Leakage-resistant model selection and held-out evaluation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ID_COLUMN, TARGET_COLUMN

RANDOM_STATE = 7


@dataclass(frozen=True)
class ExperimentResult:
    cv_results: pd.DataFrame
    test_metrics: dict[str, float | str]
    test_predictions: pd.DataFrame
    fitted_pipeline: Pipeline


def split_features(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the holdout before any data-dependent preprocessing is fitted."""

    features = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    target = frame[TARGET_COLUMN].astype(int)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def _column_groups(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(frame: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing whose learned statistics are fitted within CV/training folds."""

    numeric_columns, categorical_columns = _column_groups(frame)
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=True,
    )


def candidate_models(training_features: pd.DataFrame) -> dict[str, Pipeline]:
    """Return the two models demonstrated in the original notebook."""

    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", build_preprocessor(training_features, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", build_preprocessor(training_features, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=20,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def run_experiment(
    frame: pd.DataFrame,
    *,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> ExperimentResult:
    """Select by training-only CV, then evaluate once on the untouched holdout."""

    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    x_train, x_test, y_train, y_test = split_features(
        frame,
        test_size=test_size,
        random_state=random_state,
    )
    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    rows: list[dict[str, float | str | int]] = []
    models = candidate_models(x_train)
    for name, pipeline in models.items():
        scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            scoring="roc_auc",
            cv=cross_validation,
            n_jobs=1,
        )
        rows.append(
            {
                "model": name,
                "cv_folds": cv_folds,
                "mean_roc_auc": float(scores.mean()),
                "std_roc_auc": float(scores.std(ddof=1)),
            }
        )

    cv_results = (
        pd.DataFrame(rows).sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
    )
    selected_name = str(cv_results.loc[0, "model"])
    selected = models[selected_name]
    selected.fit(x_train, y_train)

    probabilities = selected.predict_proba(x_test)[:, 1]
    predictions = selected.predict(x_test)
    test_metrics: dict[str, float | str] = {
        "selected_model": selected_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    prediction_frame = pd.DataFrame(
        {
            "row_index": x_test.index,
            "actual_churn": y_test.to_numpy(),
            "churn_probability": probabilities,
        }
    )
    return ExperimentResult(cv_results, test_metrics, prediction_frame, selected)


def risk_deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize held-out ranking quality in equal-sized risk groups."""

    required = {"actual_churn", "churn_probability"}
    if missing := required - set(predictions.columns):
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    ranked = (
        predictions.sort_values("churn_probability", ascending=False).reset_index(drop=True).copy()
    )
    ranked["risk_decile"] = pd.qcut(
        ranked["churn_probability"].rank(method="first", ascending=False),
        10,
        labels=range(1, 11),
    )
    baseline = float(ranked["actual_churn"].mean())
    total_churners = int(ranked["actual_churn"].sum())
    table = (
        ranked.groupby("risk_decile", observed=False)
        .agg(
            customers=("actual_churn", "size"),
            actual_churners=("actual_churn", "sum"),
            actual_churn_rate=("actual_churn", "mean"),
            mean_score=("churn_probability", "mean"),
        )
        .reset_index()
    )
    table["cumulative_churners_captured"] = table["actual_churners"].cumsum() / total_churners
    table["lift_vs_sample_baseline"] = table["actual_churn_rate"] / baseline
    return table


def hypothetical_campaign(
    deciles: pd.DataFrame,
    *,
    targeted_deciles: int,
    contact_cost: float,
    retention_success_rate: float,
    annual_customer_value: float,
) -> dict[str, float | int | str]:
    """Calculate a labeled scenario; it is not an estimate of achieved causal impact."""

    if targeted_deciles not in range(1, 11):
        raise ValueError("targeted_deciles must be between 1 and 10")
    if contact_cost < 0 or annual_customer_value < 0:
        raise ValueError("Monetary assumptions cannot be negative")
    if not 0 <= retention_success_rate <= 1:
        raise ValueError("retention_success_rate must be between 0 and 1")

    targeted = deciles.iloc[:targeted_deciles]
    contacts = int(targeted["customers"].sum())
    observed_churners = int(targeted["actual_churners"].sum())
    campaign_cost = contacts * contact_cost
    hypothetical_saved = observed_churners * retention_success_rate
    protected_value = hypothetical_saved * annual_customer_value
    net_value = protected_value - campaign_cost
    return {
        "status": "hypothetical; requires a randomized campaign to estimate causal lift",
        "contacts": contacts,
        "observed_churners_in_holdout": observed_churners,
        "assumed_retention_success_rate": retention_success_rate,
        "campaign_cost": campaign_cost,
        "hypothetical_protected_value": protected_value,
        "hypothetical_net_value": net_value,
    }
