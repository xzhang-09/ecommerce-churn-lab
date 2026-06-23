from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deployment_components_are_removed():
    removed_paths = [
        "dockerfile",
        ".dockerignore",
        ".github",
        "src/app",
        "src/serving",
        "scripts/test_fastapi.py",
    ]

    for relative_path in removed_paths:
        assert not (ROOT / relative_path).exists()


def test_core_analysis_modules_import():
    from src.data.load_data import load_data
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features
    from src.models.tune import tune_model

    assert callable(load_data)
    assert callable(preprocess_data)
    assert callable(build_features)
    assert callable(tune_model)
