"""Dataset loading and integrity checks."""

from pathlib import Path

import pandas as pd

ID_COLUMN = "Customer_ID"
TARGET_COLUMN = "churn"
EXPECTED_FILENAMES = ("Client.csv", "Record.csv")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required dataset file not found: {path}. "
            "See data/README.md for the manual placement instructions."
        )
    return pd.read_csv(path)


def _validate_identifier(frame: pd.DataFrame, name: str) -> None:
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"{name} is missing required column {ID_COLUMN!r}")
    if frame[ID_COLUMN].isna().any():
        raise ValueError(f"{name}.{ID_COLUMN} contains null values")
    duplicate_count = int(frame[ID_COLUMN].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{name}.{ID_COLUMN} contains {duplicate_count} duplicate rows")


def load_telecom_data(data_dir: str | Path) -> pd.DataFrame:
    """Load and one-to-one merge the course-provided Client and Record tables.

    The function deliberately rejects missing, duplicate, or unmatched identifiers so a
    silent many-to-many merge cannot change the analytical population.
    """

    directory = Path(data_dir)
    client = _read_csv(directory / EXPECTED_FILENAMES[0])
    record = _read_csv(directory / EXPECTED_FILENAMES[1])

    _validate_identifier(client, "Client.csv")
    _validate_identifier(record, "Record.csv")

    client_ids = set(client[ID_COLUMN])
    record_ids = set(record[ID_COLUMN])
    client_only = len(client_ids - record_ids)
    record_only = len(record_ids - client_ids)
    if client_only or record_only:
        raise ValueError(
            "Client.csv and Record.csv contain unmatched Customer_ID values: "
            f"client_only={client_only}, record_only={record_only}"
        )

    merged = record.merge(
        client,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if TARGET_COLUMN not in merged.columns:
        raise ValueError(f"Merged data is missing required target column {TARGET_COLUMN!r}")
    if merged[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN!r} contains null values")

    observed_targets = set(merged[TARGET_COLUMN].unique())
    if not observed_targets.issubset({0, 1}):
        raise ValueError(
            f"{TARGET_COLUMN!r} must be binary 0/1; observed {sorted(observed_targets)!r}"
        )
    if merged[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"{TARGET_COLUMN!r} must contain both classes")

    return merged


def profile_data(frame: pd.DataFrame) -> dict[str, object]:
    """Return compact shape, target, missingness, and cardinality evidence."""

    feature_frame = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    categorical = feature_frame.select_dtypes(include=["object", "category", "string"])
    missing = frame.isna().mean().sort_values(ascending=False)
    cardinality = categorical.nunique(dropna=False).sort_values(ascending=False)
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "target_counts": {
            str(key): int(value) for key, value in frame[TARGET_COLUMN].value_counts().items()
        },
        "top_missing_percent": {
            str(key): round(float(value * 100), 3)
            for key, value in missing[missing.gt(0)].head(15).items()
        },
        "top_categorical_cardinality": {
            str(key): int(value) for key, value in cardinality.head(15).items()
        },
    }
