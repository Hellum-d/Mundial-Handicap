"""Dynamic Elo rating system for international football.

A pragmatic, self-updating rating model. Ratings are walked forward through
the match history in chronological order so there is never any future leakage:
the rating used to "predict" a match only ever reflects matches that preceded
it. The K-factor scales with match importance (a friendly nudges ratings; a
World Cup final moves them a lot) and the rating update is dampened by a
goal-margin multiplier (the Clark modification) so that a 5-0 thrashing does
not over-inflate ratings relative to a 1-0 win.

Outcome probabilities are derived from the rating difference with a simple,
monotone draw model. Elo is primarily a strong, well-calibrated baseline and a
second opinion for the ensemble; the Dixon-Coles model is the scoreline engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor import config
from football_predictor.models.base_model import BaseModel, OutcomeProba


def _k_factor(match_weight: float) -> float:
    """Return the Elo K-factor for a match of the given importance weight.

    Falls back to the nearest tabulated weight when an exact key is absent.
    """
    table = config.ELO_K_BY_WEIGHT
    if match_weight in table:
        return table[match_weight]
    nearest = min(table, key=lambda w: abs(w - match_weight))
    return table[nearest]


def _margin_multiplier(goal_diff: int, winner_rating_diff: float) -> float:
    """Clark goal-margin multiplier.

    Scales the rating update by the goal margin while damping the effect when
    the favourite wins (a strong team beating a weak one by many goals should
    not swing ratings as hard as an upset of the same margin).

    Args:
        goal_diff: Absolute goal difference of the match (>= 1 for a result).
        winner_rating_diff: ``winner_elo - loser_elo`` *before* the match.
    """
    if goal_diff <= 0:
        return 1.0
    return float(
        np.log(abs(goal_diff) + 1.0) * (2.2 / (winner_rating_diff * 0.001 + 2.2))
    )


class EloRatingSystem(BaseModel):
    """Walk-forward Elo ratings with importance- and margin-aware updates."""

    def __init__(
        self,
        initial_rating: float = config.ELO_INITIAL,
        home_advantage: float = config.ELO_HOME_ADVANTAGE,
        draw_base: float = 0.30,
    ) -> None:
        """Initialise an unfitted Elo system.

        Args:
            initial_rating: Rating assigned to a team on its first appearance.
            home_advantage: Rating points added to the non-neutral home side.
            draw_base: Peak draw probability for an evenly matched fixture; the
                draw model interpolates from this down to ~0 as the rating gap
                grows.
        """
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.draw_base = draw_base
        self.ratings: dict[str, float] = {}

    # -- fitting -----------------------------------------------------------
    def _rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)

    def fit(self, matches: pd.DataFrame) -> "EloRatingSystem":
        """Walk the match history forward, updating ratings in date order."""
        self.ratings = {}
        ordered = matches.sort_values("date").itertuples(index=False)
        for m in ordered:
            ra = self._rating(m.team_a)
            rb = self._rating(m.team_b)

            home_adv = 0.0 if bool(getattr(m, "neutral", True)) else self.home_advantage
            ra_eff = ra + home_adv

            # Expected score for A (includes half-weight of draws).
            exp_a = 1.0 / (1.0 + 10.0 ** (-(ra_eff - rb) / 400.0))

            ga, gb = int(m.goals_a), int(m.goals_b)
            score_a = 1.0 if ga > gb else 0.5 if ga == gb else 0.0

            k = _k_factor(float(m.match_weight))
            if ga != gb:
                winner_diff = (ra_eff - rb) if ga > gb else (rb - ra_eff)
                mult = _margin_multiplier(abs(ga - gb), winner_diff)
            else:
                mult = 1.0

            delta = k * mult * (score_a - exp_a)
            self.ratings[m.team_a] = ra + delta
            self.ratings[m.team_b] = rb - delta

        self.is_fitted = True
        return self

    # -- prediction --------------------------------------------------------
    def expected_score(self, team_a: str, team_b: str, *, neutral: bool = True) -> float:
        """Elo expected score for A in ``[0, 1]`` (draws count as a half)."""
        self._check_fitted()
        home_adv = 0.0 if neutral else self.home_advantage
        diff = (self._rating(team_a) + home_adv) - self._rating(team_b)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def predict_proba(
        self, team_a: str, team_b: str, *, neutral: bool = True
    ) -> OutcomeProba:
        """Win/draw/win probabilities from the rating gap.

        The draw probability peaks for evenly matched sides and decays linearly
        in the gap; the remaining mass is split between the two sides in
        proportion to the Elo expected score.
        """
        self._check_fitted()
        w = self.expected_score(team_a, team_b, neutral=neutral)
        # Even match (w=0.5) -> max draw; lopsided (w->0 or 1) -> ~no draw.
        p_draw = self.draw_base * (1.0 - 2.0 * abs(w - 0.5))
        p_draw = max(0.0, p_draw)
        remaining = 1.0 - p_draw
        return {
            "win_a": remaining * w,
            "draw": p_draw,
            "win_b": remaining * (1.0 - w),
        }
