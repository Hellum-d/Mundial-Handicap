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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from football_predictor import config
from football_predictor.evaluation import metrics
from football_predictor.models.elo_model import EloRatingSystem
from football_predictor.output.prediction_engine import PredictionEngine


@dataclass
class FoldResult:
    """Metrics for one tournament fold across all evaluated systems."""

    test_label: str
    n_matches: int
    scores: dict[str, dict[str, float]]  # system -> {log_loss, brier, rps}
    blend_w_mc: float = float("nan")  # fitted Monte-Carlo/Dixon-Coles weight
    draw_gamma: float = float("nan")  # fitted draw-probability multiplier
    # system -> per-match log loss vector (for pooled bootstrap CIs).
    per_match_log_loss: dict[str, np.ndarray] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"[{self.test_label}]  ({self.n_matches} matches"
            f", blend w_mc={self.blend_w_mc:.2f}, draw_gamma={self.draw_gamma:.2f})"
        ]
        for system, m in self.scores.items():
            lines.append(
                f"  {system:<12} "
                f"log_loss={m['log_loss']:.4f}  "
                f"brier={m['brier']:.4f}  "
                f"rps={m['rps']:.4f}"
            )
        return "\n".join(lines)


# Tournaments with fewer matches than this are treated as in-progress /
# incomplete and skipped as test folds (e.g. an ongoing World Cup).
_MIN_FOLD_MATCHES = 16


def _score(probs: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return {
        "log_loss": metrics.log_loss(probs, truth),
        "brier": metrics.brier_score(probs, truth),
        "rps": metrics.ranked_probability_score(probs, truth),
    }


class Backtester:
    """Run temporal-CV folds and report metrics vs. baselines."""

    def __init__(
        self,
        matches: pd.DataFrame,
        test_competition: str = "world_cup",
        train_window_years: int | None = None,
        min_test_year: int | None = None,
        validation_months: int = config.BACKTEST_VALIDATION_MONTHS,
        calibrate_draw: bool = False,
    ) -> None:
        """Initialise with the full match dataset.

        Args:
            matches: All historical matches with a ``competition`` column.
            test_competition: Competition value whose tournaments form the
                held-out test sets (one fold per distinct year).
            train_window_years: If set, each fold trains only on matches within
                this many years before the tested tournament (the exponential
                time-decay makes older matches negligible anyway, and it keeps
                the fit tractable on the full 49k-row real dataset). ``None``
                trains on all prior matches.
            min_test_year: If set, only tournaments from this year onward are
                evaluated as folds (the modern, data-rich era).
            validation_months: Length of the leak-free validation slice carved
                out immediately before each test tournament (see ``split_fold``).
            calibrate_draw: Whether to fit the post-hoc draw multiplier on the
                validation slice. **Off by default**: on this data it hurts the
                held-out metric (mean log loss 1.0038 -> 1.0077) because the
                pre-tournament validation matches (qualifiers/friendlies) have a
                different draw distribution than World Cup matches, so the fitted
                gamma generalises poorly. Kept available for when a World-Cup-like
                validation set exists.
        """
        self.matches = matches.copy()
        self.matches["date"] = pd.to_datetime(self.matches["date"])
        self.matches["year"] = self.matches["date"].dt.year
        self.test_competition = test_competition
        self.train_window_years = train_window_years
        self.min_test_year = min_test_year
        self.validation_months = validation_months
        self.calibrate_draw = calibrate_draw

    def test_years(self) -> list[int]:
        """Years with a *complete* tournament of the test competition.

        Skips in-progress tournaments (fewer than ``_MIN_FOLD_MATCHES`` matches)
        and, if ``min_test_year`` is set, anything before it.
        """
        mask = self.matches["competition"] == self.test_competition
        counts = self.matches.loc[mask].groupby("year").size()
        years = [int(y) for y, n in counts.items() if n >= _MIN_FOLD_MATCHES]
        if self.min_test_year is not None:
            years = [y for y in years if y >= self.min_test_year]
        return sorted(years)

    def split_fold(
        self, year: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Leak-free ``(train, validation, test)`` split for one fold.

        Partitioned strictly by date so nothing downstream can leak the test
        set into model fitting or calibration:

          * ``test``       = the tested tournament (this competition, this year);
          * ``validation`` = all matches in the ``validation_months`` window
            immediately *before* the test tournament — used to fit calibration
            or blend weights, and disjoint from both train and test;
          * ``train``      = matches before the validation window (and within
            ``train_window_years`` of the test, if set).

        Invariant: ``max(train.date) < min(validation.date)`` and
        ``max(validation.date) < min(test.date)``.
        """
        is_test = (self.matches["competition"] == self.test_competition) & (
            self.matches["year"] == year
        )
        test = self.matches[is_test]
        if test.empty:
            raise ValueError(f"No {self.test_competition} tournament in {year}.")

        test_start = test["date"].min()
        val_start = test_start - pd.DateOffset(months=self.validation_months)
        before_test = self.matches[self.matches["date"] < test_start]

        validation = before_test[before_test["date"] >= val_start]
        train = before_test[before_test["date"] < val_start]
        if self.train_window_years is not None:
            window_start = test_start - pd.DateOffset(years=self.train_window_years)
            train = train[train["date"] >= window_start]
        return train, validation, test

    def run(self) -> list[FoldResult]:
        """Execute every fold and return per-fold metric breakdowns."""
        results: list[FoldResult] = []
        for year in self.test_years():
            results.append(self._run_fold(year))
        return results

    def _run_fold(self, year: int) -> FoldResult:
        train, validation, test = self.split_fold(year)
        full = pd.concat([train, validation])

        # Leak-free calibration: base models see only `train`; the blend weight
        # (and optionally the draw multiplier) are fitted on the disjoint
        # `validation` slice.
        probe = PredictionEngine().fit(train).fit_blend(validation)
        if self.calibrate_draw:
            probe.fit_draw_calibration(validation)
        # Final models use *all* pre-test data; reuse the learned parameters.
        engine = PredictionEngine(
            w_monte_carlo=probe.w_mc, w_elo=probe.w_elo, draw_gamma=probe.draw_gamma
        ).fit(full)
        elo_only = EloRatingSystem().fit(full)

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

        systems = {"ensemble": ens_probs, "elo_only": elo_probs, "uniform": uniform}
        return FoldResult(
            test_label=f"{self.test_competition} {year}",
            n_matches=len(truth),
            scores={k: _score(p, truth) for k, p in systems.items()},
            blend_w_mc=engine.w_mc,
            draw_gamma=engine.draw_gamma,
            per_match_log_loss={
                k: metrics.log_loss_per_match(p, truth) for k, p in systems.items()
            },
        )


def significance_report(
    folds: list[FoldResult], n_boot: int = 10_000, seed: int = 0
) -> str:
    """Pool per-match log loss across folds and bootstrap CIs for significance.

    Reports each system's pooled log loss with a 90% bootstrap CI, then the
    paired ``ensemble - baseline`` difference with its CI. A wholly-negative
    difference interval means the ensemble is significantly better than that
    baseline (log loss: lower is better) — the direct answer to "is this gap
    real or just noise on ~250 matches?".
    """
    if not folds:
        return "No folds to report."
    systems = ("ensemble", "elo_only", "uniform")
    pooled = {
        s: np.concatenate([f.per_match_log_loss[s] for f in folds]) for s in systems
    }
    n = len(pooled["ensemble"])

    lines = [f"=== Pooled log loss, 90% bootstrap CI (n={n} matches) ==="]
    for s in systems:
        mean, lo, hi = metrics.bootstrap_ci(pooled[s], n_boot=n_boot, seed=seed)
        lines.append(f"  {s:<12} {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    lines.append("=== ensemble - baseline (negative = ensemble better) ===")
    for base in ("elo_only", "uniform"):
        diff, lo, hi = metrics.bootstrap_diff_ci(
            pooled["ensemble"], pooled[base], n_boot=n_boot, seed=seed
        )
        verdict = "significant" if hi < 0 else ("worse" if lo > 0 else "n.s.")
        lines.append(
            f"  ensemble - {base:<9} {diff:+.4f}  [{lo:+.4f}, {hi:+.4f}]  ({verdict})"
        )
    return "\n".join(lines)
