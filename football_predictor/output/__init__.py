"""Output layer: validated prediction schema and assembly engine."""

from football_predictor.output.schemas import MatchPrediction
from football_predictor.output.prediction_engine import (
    PredictionEngine,
    load_or_train_engine,
)

__all__ = ["MatchPrediction", "PredictionEngine", "load_or_train_engine"]
