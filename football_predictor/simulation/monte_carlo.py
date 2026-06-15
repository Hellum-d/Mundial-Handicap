"""Vectorised Monte-Carlo match simulator.

Given the two scoring rates ``(lambda_A, lambda_B)`` from the scoreline model,
this samples ``N`` independent matches with NumPy Poisson draws (no Python-level
loops over simulations) and reduces them into outcome probabilities, a
scoreline distribution, secondary betting markets, and goal confidence
intervals.

For knockout fixtures a draw after 90 minutes is resolved with a short
extra-time period (a fraction of a match, same rates) and then a coin-flip
penalty shootout, so the reported probabilities are conditional win
probabilities with no draw mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from football_predictor import config


@dataclass
class MonteCarloResult:
    """Reduced output of a Monte-Carlo match simulation.

    Probabilities are over the regulation-time result for group fixtures; for
    knockout fixtures ``draw`` is 0 and the win probabilities include
    extra-time/penalty resolution.
    """

    win_a: float
    draw: float
    win_b: float
    xg_a: float
    xg_b: float
    # 90% *prediction* interval on the realized goal count (Poisson sampling
    # spread), not a confidence interval on the expected-goals parameter.
    goals_a_pi_90: tuple[float, float]
    goals_b_pi_90: tuple[float, float]
    over_2_5: float
    btts: float
    p_team_a_scores: float
    p_team_b_scores: float
    clean_sheet_a: float
    clean_sheet_b: float
    top_scorelines: list[dict] = field(default_factory=list)

    def outcome_proba(self) -> dict[str, float]:
        """Return the ``{win_a, draw, win_b}`` triple as a dict."""
        return {"win_a": self.win_a, "draw": self.draw, "win_b": self.win_b}


class MonteCarloEngine:
    """Sample match outcomes from Poisson scoring rates, fully vectorised."""

    def __init__(
        self,
        n_sims: int = config.MC_DEFAULT_SIMS,
        seed: int = config.RANDOM_SEED,
        extra_time_fraction: float = config.EXTRA_TIME_FRACTION,
    ) -> None:
        """Configure the engine.

        Args:
            n_sims: Number of simulated matches per ``run`` call.
            seed: Seed for the NumPy random generator (reproducibility).
            extra_time_fraction: Length of extra time as a fraction of a match,
                used to scale the scoring rates in knockout tie-breaks.
        """
        self.n_sims = n_sims
        self.extra_time_fraction = extra_time_fraction
        self.rng = np.random.default_rng(seed)

    def run(
        self, lambda_a: float, lambda_b: float, *, stage: str = "group"
    ) -> MonteCarloResult:
        """Simulate a fixture and reduce to probabilities and markets.

        Args:
            lambda_a: Expected goals for team A.
            lambda_b: Expected goals for team B.
            stage: ``"group"`` keeps draws; anything else is treated as a
                knockout and resolves draws via extra time and penalties.
        """
        n = self.n_sims
        goals_a = self.rng.poisson(lam=lambda_a, size=n)
        goals_b = self.rng.poisson(lam=lambda_b, size=n)

        # Markets and scoreline distribution use regulation-time goals.
        over_2_5 = float(np.mean((goals_a + goals_b) > 2.5))
        btts = float(np.mean((goals_a > 0) & (goals_b > 0)))
        p_a_scores = float(np.mean(goals_a > 0))
        p_b_scores = float(np.mean(goals_b > 0))
        clean_sheet_a = float(np.mean(goals_b == 0))
        clean_sheet_b = float(np.mean(goals_a == 0))

        goals_a_pi = (
            float(np.percentile(goals_a, 5)),
            float(np.percentile(goals_a, 95)),
        )
        goals_b_pi = (
            float(np.percentile(goals_b, 5)),
            float(np.percentile(goals_b, 95)),
        )

        top_scorelines = self._top_scorelines(goals_a, goals_b)

        is_knockout = stage != "group"
        if not is_knockout:
            win_a = float(np.mean(goals_a > goals_b))
            draw = float(np.mean(goals_a == goals_b))
            win_b = float(np.mean(goals_a < goals_b))
        else:
            a_wins = goals_a > goals_b
            b_wins = goals_a < goals_b
            draw_mask = goals_a == goals_b
            self._resolve_knockout(
                a_wins, b_wins, draw_mask, lambda_a, lambda_b
            )
            win_a = float(np.mean(a_wins))
            win_b = float(np.mean(b_wins))
            draw = 0.0

        return MonteCarloResult(
            win_a=win_a,
            draw=draw,
            win_b=win_b,
            xg_a=float(np.mean(goals_a)),
            xg_b=float(np.mean(goals_b)),
            goals_a_pi_90=goals_a_pi,
            goals_b_pi_90=goals_b_pi,
            over_2_5=over_2_5,
            btts=btts,
            p_team_a_scores=p_a_scores,
            p_team_b_scores=p_b_scores,
            clean_sheet_a=clean_sheet_a,
            clean_sheet_b=clean_sheet_b,
            top_scorelines=top_scorelines,
        )

    # -- internals ---------------------------------------------------------
    def _resolve_knockout(
        self,
        a_wins: np.ndarray,
        b_wins: np.ndarray,
        draw_mask: np.ndarray,
        lambda_a: float,
        lambda_b: float,
    ) -> None:
        """In place: assign every drawn sim a winner via extra time + pens."""
        n_draw = int(draw_mask.sum())
        if n_draw == 0:
            return
        et_a = self.rng.poisson(lambda_a * self.extra_time_fraction, size=n_draw)
        et_b = self.rng.poisson(lambda_b * self.extra_time_fraction, size=n_draw)

        draw_idx = np.flatnonzero(draw_mask)
        et_a_wins = et_a > et_b
        et_b_wins = et_a < et_b
        still_level = et_a == et_b

        a_wins[draw_idx[et_a_wins]] = True
        b_wins[draw_idx[et_b_wins]] = True

        # Penalty shootout: fair coin for whatever is still level.
        level_idx = draw_idx[still_level]
        if level_idx.size:
            pen_a = self.rng.random(level_idx.size) < 0.5
            a_wins[level_idx[pen_a]] = True
            b_wins[level_idx[~pen_a]] = True

    @staticmethod
    def _top_scorelines(
        goals_a: np.ndarray, goals_b: np.ndarray, top_k: int = 5, cap: int = 8
    ) -> list[dict]:
        """Return the ``top_k`` most likely scorelines as score/probability."""
        n = goals_a.size
        ca = np.clip(goals_a, 0, cap)
        cb = np.clip(goals_b, 0, cap)
        pairs = np.stack([ca, cb], axis=1)
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1][:top_k]
        return [
            {"score": f"{int(uniq[i, 0])}-{int(uniq[i, 1])}",
             "probability": round(float(counts[i] / n), 4)}
            for i in order
        ]
