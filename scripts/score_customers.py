#!/usr/bin/env python3
"""Thin CLI entry point for batch-scoring customers with a trained churn model.

Logic lives in :mod:`churn.models.score`; equivalent to the `churn-score`
console script created on install.
"""

from churn.models.score import main

if __name__ == "__main__":
    main()
