import json
from pathlib import Path


def test_eda_notebook_does_not_duplicate_formal_model_training():
    notebook = json.loads(Path("notebooks/EDA.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    disallowed_training_imports = [
        "import lightgbm",
        "from lightgbm",
        "import optuna",
        "import mlflow",
    ]

    for text in disallowed_training_imports:
        assert text not in source


def test_eda_notebook_points_to_pipeline_model_comparison_output():
    notebook = json.loads(Path("notebooks/EDA.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    assert "artifacts/model_comparison.csv" in source
    assert "scripts/run_pipeline.py" in source
