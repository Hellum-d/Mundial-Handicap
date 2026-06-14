"""Evaluation: probabilistic metrics and temporal back-testing."""

from football_predictor.evaluation.metrics import (
    brier_score,
    log_loss,
    ranked_probability_score,
)
from football_predictor.evaluation.backtester import Backtester, FoldResult

__all__ = [
    "log_loss",
    "brier_score",
    "ranked_probability_score",
    "Backtester",
    "FoldResult",
]
