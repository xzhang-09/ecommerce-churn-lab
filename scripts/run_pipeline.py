#!/usr/bin/env python3
"""Thin CLI entry point for the churn modeling pipeline.

All logic lives in the importable package (:mod:`churn.pipeline`); this wrapper
exists so the documented `python scripts/run_pipeline.py ...` invocation keeps
working. Equivalent to the `churn-pipeline` console script created on install.
"""

from churn.pipeline import cli

if __name__ == "__main__":
    cli()
