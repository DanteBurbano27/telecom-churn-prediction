import numpy as np
import pandas as pd

from telecom_churn.modeling import (
    build_preprocessor,
    hypothetical_campaign,
    risk_deciles,
    run_experiment,
)


def _synthetic_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    signal = rng.normal(size=rows)
    category = np.where(signal > 0, "high", "low").astype(object)
    category[::17] = None
    numeric = signal + rng.normal(scale=0.7, size=rows)
    numeric[::19] = np.nan
    target = (signal + rng.normal(scale=0.9, size=rows) > 0).astype(int)
    return pd.DataFrame(
        {
            "Customer_ID": np.arange(1, rows + 1),
            "numeric_signal": numeric,
            "segment": category,
            "churn": target,
        }
    )


def test_preprocessor_learns_numeric_imputation_from_training_data_only() -> None:
    training = pd.DataFrame({"numeric": [1.0, 2.0, np.nan], "category": ["a", "b", None]})
    preprocessor = build_preprocessor(training, scale_numeric=False)

    preprocessor.fit(training)

    numeric_imputer = preprocessor.named_transformers_["numeric"].named_steps["impute"]
    assert numeric_imputer.statistics_[0] == 1.5


def test_experiment_uses_training_cv_and_returns_holdout_predictions() -> None:
    frame = _synthetic_frame()

    result = run_experiment(frame, cv_folds=3, test_size=0.25)

    assert set(result.cv_results["model"]) == {"Logistic Regression", "Random Forest"}
    assert set(result.cv_results["cv_folds"]) == {3}
    assert len(result.test_predictions) == 60
    assert 0 <= float(result.test_metrics["roc_auc"]) <= 1


def test_deciles_and_hypothetical_campaign_are_explicit() -> None:
    predictions = pd.DataFrame(
        {
            "actual_churn": [1, 0] * 50,
            "churn_probability": np.linspace(0.99, 0.01, 100),
        }
    )

    deciles = risk_deciles(predictions)
    scenario = hypothetical_campaign(
        deciles,
        targeted_deciles=2,
        contact_cost=10,
        retention_success_rate=0.25,
        annual_customer_value=100,
    )

    assert deciles["customers"].sum() == 100
    assert scenario["contacts"] == 20
    assert str(scenario["status"]).startswith("hypothetical")
