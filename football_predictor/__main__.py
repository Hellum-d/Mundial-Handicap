"""Command-line demo for the prediction engine.

Run with::

    python -m football_predictor

It generates (or loads) the synthetic dataset, runs the temporal back-test
against the uniform and Elo-only baselines, and prints one fully-assembled
example prediction.
"""

from __future__ import annotations

import pandas as pd

from football_predictor import config
from football_predictor.data.real_data import load_real_matches
from football_predictor.evaluation.backtester import Backtester
from football_predictor.output.prediction_engine import PredictionEngine


def main() -> None:
    matches = load_real_matches()
    print(f"Loaded {len(matches)} real matches "
          f"({matches['date'].min().date()} -> {matches['date'].max().date()})\n")

    print("=== Temporal back-test (held-out World Cups) ===")
    folds = Backtester(
        matches,
        train_window_years=config.TRAIN_WINDOW_YEARS,
        min_test_year=2010,
    ).run()
    for fold in folds:
        print(fold)

    # Average ensemble log loss across folds vs. baselines.
    systems = ["ensemble", "elo_only", "uniform"]
    print("\n=== Mean log loss across folds ===")
    for system in systems:
        mean_ll = sum(f.scores[system]["log_loss"] for f in folds) / len(folds)
        print(f"  {system:<12} {mean_ll:.4f}")

    print("\n=== Example prediction: Brazil vs Argentina (final) ===")
    # Fit the example engine on the most recent window for speed and relevance.
    recent_cutoff = matches["date"].max() - pd.DateOffset(
        years=config.TRAIN_WINDOW_YEARS
    )
    recent = matches[matches["date"] >= recent_cutoff]
    engine = PredictionEngine().fit(recent)
    pred = engine.predict("Brazil", "Argentina", stage="final", neutral=True)
    print(f"  win {pred.team_a}: {pred.win_probability_a:.3f}")
    print(f"  draw          : {pred.draw_probability:.3f}")
    print(f"  win {pred.team_b}: {pred.win_probability_b:.3f}")
    print(f"  xG: {pred.xg_a:.2f} (90% CI {pred.xg_a_ci_90}) vs "
          f"{pred.xg_b:.2f} (90% CI {pred.xg_b_ci_90})")
    print(f"  over 2.5: {pred.over_2_5_probability:.3f}  "
          f"BTTS: {pred.btts_probability:.3f}")
    print(f"  top scorelines: {pred.top_scorelines}")
    print(f"  model agreement: {pred.model_agreement_score:.3f}  "
          f"confidence: {pred.confidence_score:.3f}")


if __name__ == "__main__":
    main()
