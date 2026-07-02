from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_deployment_components_are_removed():
    # Guards against the *serving* infrastructure creeping back (the project is
    # scoped to offline analysis). Note: `.github` is intentionally NOT here —
    # CI is offline tooling, not a serving component, so it is allowed.
    removed_paths = [
        "dockerfile",
        ".dockerignore",
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
    from churn.features.build_features import apply_feature_schema, build_feature_schema
    from churn.models.score import score_dataframe
    from churn.models.tune import tune_model
    from churn.validation import validate_data

    assert callable(load_data)
    assert callable(preprocess_data)
    assert callable(build_feature_schema)
    assert callable(apply_feature_schema)
    assert callable(score_dataframe)
    assert callable(tune_model)
    assert callable(validate_data.validate_ecommerce_data)
