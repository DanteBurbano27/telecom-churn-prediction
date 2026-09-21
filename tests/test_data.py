from pathlib import Path

import pandas as pd
import pytest

from telecom_churn.data import load_telecom_data, profile_data


def _write_tables(directory: Path, *, duplicate_client: bool = False) -> None:
    client = pd.DataFrame(
        {
            "Customer_ID": [1, 2, 3, 4],
            "segment": ["A", "A", "B", None],
            "tenure": [2, 8, 4, 10],
        }
    )
    if duplicate_client:
        client.loc[3, "Customer_ID"] = 3
    record = pd.DataFrame(
        {
            "Customer_ID": [1, 2, 3, 4],
            "usage": [1.0, 2.0, None, 4.0],
            "churn": [0, 1, 0, 1],
        }
    )
    client.to_csv(directory / "Client.csv", index=False)
    record.to_csv(directory / "Record.csv", index=False)


def test_load_telecom_data_validates_and_merges_one_to_one(tmp_path: Path) -> None:
    _write_tables(tmp_path)

    merged = load_telecom_data(tmp_path)
    profile = profile_data(merged)

    assert merged.shape == (4, 5)
    assert profile["target_counts"] == {"0": 2, "1": 2}
    assert profile["top_missing_percent"] == {"usage": 25.0, "segment": 25.0}


def test_load_telecom_data_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    _write_tables(tmp_path, duplicate_client=True)

    with pytest.raises(ValueError, match="duplicate"):
        load_telecom_data(tmp_path)


def test_load_telecom_data_rejects_unmatched_identifiers(tmp_path: Path) -> None:
    _write_tables(tmp_path)
    record = pd.read_csv(tmp_path / "Record.csv")
    record.loc[3, "Customer_ID"] = 5
    record.to_csv(tmp_path / "Record.csv", index=False)

    with pytest.raises(ValueError, match="unmatched"):
        load_telecom_data(tmp_path)
