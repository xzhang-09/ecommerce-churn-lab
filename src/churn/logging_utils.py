"""Lightweight logging setup for the churn package.

The pipeline used to communicate progress with bare ``print`` calls. Those are
now routed through the standard ``logging`` module so output can be silenced,
redirected, or captured without touching the code that emits it.

Only the ``churn`` package logger is configured here — third-party libraries
(mlflow, optuna, lightgbm) keep their own logging config, so this does not make
them noisier. Library modules just call ``logging.getLogger(__name__)``; the
CLI entry points call :func:`configure_logging` once so messages actually reach
the console with a plain, print-like format.
"""

import logging

LOGGER_NAME = "churn"


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a plain-message stream handler to the ``churn`` logger.

    Idempotent: safe to call more than once (e.g. once per pipeline run in a
    test session) without stacking duplicate handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(h, "_churn_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._churn_handler = True  # marker so we don't add it twice
        logger.addHandler(handler)
    logger.setLevel(level)
    # Don't also propagate to the root logger, or messages print twice if the
    # host application has configured root logging.
    logger.propagate = False
