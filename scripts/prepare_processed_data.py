#!/usr/bin/env python3
"""Thin CLI entry point for preparing processed data without model training.

Logic lives in :mod:`churn.data.prepare`; equivalent to the `churn-prepare`
console script created on install.
"""

from churn.data.prepare import main

if __name__ == "__main__":
    main()
