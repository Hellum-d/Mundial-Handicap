"""Instant single-match prediction from the cached engine.

The first invocation fits and caches the engine (slow, one-off); every later
call loads the cache and predicts immediately.

Usage::

    python -m football_predictor.predict "Brazil" "Argentina" --stage final
    python -m football_predictor.predict "Spain" "Portugal" --home   # A at home
    python -m football_predictor.predict "France" "Morocco" --retrain
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from football_predictor import config
from football_predictor.data.real_data import load_real_matches
from football_predictor.output.prediction_engine import load_or_train_engine
from football_predictor.output.schemas import Stage

_STAGES = ("group", "r16", "qf", "sf", "third_place", "final")


def predict_match(
    team_a: str,
    team_b: str,
    *,
    stage: Stage = "group",
    neutral: bool = True,
    retrain: bool = False,
) -> dict:
    """Return a validated prediction as a plain dict (cached engine)."""
    matches = load_real_matches()
    cutoff = matches["date"].max() - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)
    engine = load_or_train_engine(matches[matches["date"] >= cutoff], regenerate=retrain)
    pred = engine.predict(team_a, team_b, stage=stage, neutral=neutral)
    return pred.model_dump()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single fixture.")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--stage", choices=_STAGES, default="group")
    parser.add_argument(
        "--home", action="store_true", help="team_a plays at home (non-neutral venue)"
    )
    parser.add_argument(
        "--retrain", action="store_true", help="refit and re-cache the engine"
    )
    args = parser.parse_args()

    result = predict_match(
        args.team_a,
        args.team_b,
        stage=args.stage,
        neutral=not args.home,
        retrain=args.retrain,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
