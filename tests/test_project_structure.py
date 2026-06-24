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


def test_legacy_layout_and_manual_phase_scripts_are_gone():
    # The old flat `src/<module>` layout and the time-line-named manual phase
    # scripts were consolidated into the `churn` package and pytest suites.
    legacy_paths = [
        "src/data",
        "src/features",
        "src/models",
        "src/utils",
        "src/utils/utils.py",
        "scripts/test_pipeline_phase1_data_features.py",
        "scripts/test_pipeline_phase2_modeling.py",
    ]

    for relative_path in legacy_paths:
        assert not (ROOT / relative_path).exists()


def test_packaged_under_src_layout():
    assert (ROOT / "pyproject.toml").exists()
    assert (ROOT / "src" / "churn" / "__init__.py").exists()


def test_core_analysis_modules_import():
    from churn.data.load_data import load_data
    from churn.data.preprocess import preprocess_data
    from churn.features.build_features import build_features
    from churn.models.tune import tune_model
    from churn.validation.validate_data import validate_ecommerce_data

    assert callable(load_data)
    assert callable(preprocess_data)
    assert callable(build_features)
    assert callable(tune_model)
    assert callable(validate_ecommerce_data)
