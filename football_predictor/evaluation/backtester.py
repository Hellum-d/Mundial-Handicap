"""Temporal back-testing with strict no-future-leakage splits.

Each fold trains only on matches that occurred *before* the tested tournament
and evaluates on that tournament's matches. The engine's blended probabilities
are compared against two reference baselines on the same fold:

  * uniform   - a constant (1/3, 1/3, 1/3) forecast, and
  * elo_only  - the dynamic Elo model on its own.

This makes it easy to confirm that the ensemble actually adds signal over the
naive and single-model baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_predictor.evaluation import metrics
from football_predictor.models.elo_model import EloRatingSystem
from football_predictor.output.prediction_engine import PredictionEngine


@dataclass
class FoldResult:
    """Metrics for one tournament fold across all evaluated systems."""

    test_label: str
    n_matches: int
    scores: dict[str, dict[str, float]]  # system -> {log_loss, brier, rps}

    def __str__(self) -> str:
        lines = [f"[{self.test_label}]  ({self.n_matches} matches)"]
        for system, m in self.scores.items():
            lines.append(
                f"  {system:<12} "
                f"log_loss={m['log_loss']:.4f}  "
                f"brier={m['brier']:.4f}  "
                f"rps={m['rps']:.4f}"
            )
        return "\n".join(lines)


def _score(probs: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": metrics.log_loss(probs, truth),
        "brier": metrics.brier_score(probs, truth),
        "rps": metrics.ranked_probability_score(probs, truth),
    }


class Backtester:
    """Run temporal-CV folds and report metrics vs. baselines."""

    def __init__(self, matches: pd.DataFrame, test_competition: str = "world_cup") -> None:
        """Initialise with the full match dataset.

        Args:
            matches: All historical matches with a ``competition`` column.
            test_competition: Competition value whose tournaments form the
                held-out test sets (one fold per distinct year).
        """
        self.matches = matches.copy()
        self.matches["date"] = pd.to_datetime(self.matches["date"])
        self.matches["year"] = self.matches["date"].dt.year
        self.test_competition = test_competition

    def test_years(self) -> list[int]:
        """Years that contain a tournament of the test competition."""
        mask = self.matches["competition"] == self.test_competition
        return sorted(self.matches.loc[mask, "year"].unique())

    def run(self) -> list[FoldResult]:
        """Execute every fold and return per-fold metric breakdowns."""
        results: list[FoldResult] = []
        for year in self.test_years():
            results.append(self._run_fold(year))
        return results

    def _run_fold(self, year: int) -> FoldResult:
        is_test = (self.matches["competition"] == self.test_competition) & (
            self.matches["year"] == year
        )
        test = self.matches[is_test]
        train = self.matches[self.matches["date"] < test["date"].min()]

        engine = PredictionEngine().fit(train)
        elo_only = EloRatingSystem().fit(train)

        ens_probs, elo_probs, truth = [], [], []
        for m in test.itertuples(index=False):
            neutral = bool(getattr(m, "neutral", True))
            ep = engine.predict_proba(m.team_a, m.team_b, neutral=neutral)
            lp = elo_only.predict_proba(m.team_a, m.team_b, neutral=neutral)
            ens_probs.append([ep["win_a"], ep["draw"], ep["win_b"]])
            elo_probs.append([lp["win_a"], lp["draw"], lp["win_b"]])
            truth.append(metrics.outcome_index(int(m.goals_a), int(m.goals_b)))

        ens_probs = np.array(ens_probs)
        elo_probs = np.array(elo_probs)
        truth = np.array(truth)
        uniform = np.full_like(ens_probs, 1.0 / 3.0)

        return FoldResult(
            test_label=f"{self.test_competition} {year}",
            n_matches=len(truth),
            scores={
                "ensemble": _score(ens_probs, truth),
                "elo_only": _score(elo_probs, truth),
                "uniform": _score(uniform, truth),
            },
        )
