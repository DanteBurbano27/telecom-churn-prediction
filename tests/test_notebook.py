from pathlib import Path

import nbformat


def test_primary_notebook_is_valid_and_labels_historical_metrics() -> None:
    notebook_path = Path("notebooks/01_telecom_churn_intelligence.ipynb")
    notebook = nbformat.read(notebook_path, as_version=4)

    nbformat.validate(notebook)
    markdown = "\n".join(
        "".join(cell.source) for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert "Historical saved evidence" in markdown
    assert "not corrected-pipeline results" in markdown
    assert "/content/drive" not in notebook_path.read_text(encoding="utf-8")
