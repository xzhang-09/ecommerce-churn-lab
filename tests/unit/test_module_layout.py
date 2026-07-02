"""Locks in the post-split module layout and the backward-compatible re-exports.

The reusable pieces moved out of ``churn.pipeline`` into focused modules, but
``churn.pipeline`` still re-exports them so existing imports (and downstream
code) keep working. These tests fail loudly if either the new homes or the
compatibility shims disappear.
"""

import logging

from churn import pipeline
from churn.logging_utils import LOGGER_NAME, configure_logging


def test_functions_live_in_their_new_modules():
    from churn.models.estimators import build_model, get_model_specs
    from churn.models.evaluate import metrics_at_threshold, select_threshold, threshold_free_metrics
    from churn.models.training import cross_val_oof, fit_with_early_stopping
    from churn.models.compare import compare_models

    for fn in (build_model, get_model_specs, metrics_at_threshold, select_threshold,
               threshold_free_metrics, cross_val_oof, fit_with_early_stopping, compare_models):
        assert callable(fn)


def test_pipeline_reexports_moved_names_identically():
    # Same object, not a copy — pipeline.X must be the moved function itself.
    from churn.models.evaluate import select_threshold
    from churn.models.estimators import get_model_specs
    from churn.models.compare import compare_models

    assert pipeline.select_threshold is select_threshold
    assert pipeline.get_model_specs is get_model_specs
    assert pipeline.compare_models is compare_models
    # Orchestration entry points remain on the pipeline module.
    for name in ("main", "cli", "build_arg_parser", "prepare_model_splits", "run_data_validation"):
        assert callable(getattr(pipeline, name))


def test_configure_logging_is_idempotent():
    logger = logging.getLogger(LOGGER_NAME)
    configure_logging()
    n_after_first = len(logger.handlers)
    configure_logging()
    assert len(logger.handlers) == n_after_first  # no duplicate handler stacking
