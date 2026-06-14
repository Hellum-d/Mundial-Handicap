"""Assembles a validated :class:`MatchPrediction` from the fitted models.

This is the small "meta-learner" stand-in for the MVP. It fits the two base
models once, and for each fixture:

  1. pulls the scoring rates ``(lambda_A, lambda_B)`` from Dixon-Coles,
  2. runs the Monte-Carlo engine to get a rich outcome / market distribution,
  3. blends the Monte-Carlo outcome probabilities with the Elo model's own
     outcome probabilities using the configured weights, and
  4. packages everything into the Pydantic output contract.

The blend weights and the use of two models are deliberately simple; the full
system swaps this for a stacked logistic meta-learner over seven models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor import config
from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.elo_model import EloRatingSystem
from football_predictor.output.schemas import MatchPrediction, Stage
from football_predictor.simulation.monte_carlo import MonteCarloEngine

_KNOCKOUT_STAGES = {"r16", "qf", "sf", "third_place", "final"}


def _normalise(p: dict[str, float]) -> dict[str, float]:
    total = sum(p.values())
    if total <= 0:
        return {"win_a": 1 / 3, "draw": 1 / 3, "win_b": 1 / 3}
    return {k: v / total for k, v in p.items()}


class PredictionEngine:
    """Fit base models and emit validated per-match predictions."""

    def __init__(
        self,
        n_sims: int = config.MC_DEFAULT_SIMS,
        w_monte_carlo: float = config.BLEND_WEIGHT_MONTE_CARLO,
        w_elo: float = config.BLEND_WEIGHT_ELO,
    ) -> None:
        self.dixon_coles = DixonColesModel()
        self.elo = EloRatingSystem()
        self.mc = MonteCarloEngine(n_sims=n_sims)
        self.w_mc = w_monte_carlo
        self.w_elo = w_elo
        self.is_fitted = False

    def fit(self, matches: pd.DataFrame) -> "PredictionEngine":
        """Fit both base models on the same match history."""
        self.dixon_coles.fit(matches)
        self.elo.fit(matches)
        self.is_fitted = True
        return self

    def predict_proba(
        self, team_a: str, team_b: str, *, stage: Stage = "group", neutral: bool = True
    ) -> dict[str, float]:
        """Blended ``{win_a, draw, win_b}`` outcome probabilities.

        For knockout stages the Monte-Carlo draw mass is resolved into wins and
        the Elo draw mass is split evenly before blending, so the result has no
        draw component.
        """
        lam_a, lam_b = self.dixon_coles.predict_lambdas(team_a, team_b, neutral=neutral)
        mc = self.mc.run(lam_a, lam_b, stage=stage)
        mc_p = mc.outcome_proba()
        elo_p = self.elo.predict_proba(team_a, team_b, neutral=neutral)

        if stage in _KNOCKOUT_STAGES:
            # Split Elo's draw evenly so both heads are draw-free before blending.
            half = elo_p["draw"] / 2.0
            elo_p = {
                "win_a": elo_p["win_a"] + half,
                "draw": 0.0,
                "win_b": elo_p["win_b"] + half,
            }

        blended = {
            k: self.w_mc * mc_p[k] + self.w_elo * elo_p[k]
            for k in ("win_a", "draw", "win_b")
        }
        return _normalise(blended)

    def predict(
        self, team_a: str, team_b: str, *, stage: Stage = "group", neutral: bool = True
    ) -> MatchPrediction:
        """Full validated prediction including markets and quality scores."""
        lam_a, lam_b = self.dixon_coles.predict_lambdas(team_a, team_b, neutral=neutral)
        mc = self.mc.run(lam_a, lam_b, stage=stage)
        proba = self.predict_proba(team_a, team_b, stage=stage, neutral=neutral)

        agreement = self._agreement(team_a, team_b, stage=stage, neutral=neutral)
        confidence = self._confidence(team_a, team_b)

        return MatchPrediction(
            team_a=team_a,
            team_b=team_b,
            stage=stage,
            neutral=neutral,
            win_probability_a=proba["win_a"],
            draw_probability=proba["draw"],
            win_probability_b=proba["win_b"],
            xg_a=mc.xg_a,
            xg_b=mc.xg_b,
            xg_a_ci_90=mc.xg_a_ci_90,
            xg_b_ci_90=mc.xg_b_ci_90,
            top_scorelines=mc.top_scorelines,
            over_2_5_probability=mc.over_2_5,
            btts_probability=mc.btts,
            p_team_a_scores=mc.p_team_a_scores,
            p_team_b_scores=mc.p_team_b_scores,
            clean_sheet_a=mc.clean_sheet_a,
            clean_sheet_b=mc.clean_sheet_b,
            confidence_score=confidence,
            model_agreement_score=agreement,
        )

    # -- quality scores ----------------------------------------------------
    def _agreement(
        self, team_a: str, team_b: str, *, stage: Stage, neutral: bool
    ) -> float:
        """1 - normalised total-variation distance between the two model heads.

        1.0 = the Dixon-Coles/Monte-Carlo head and the Elo head agree exactly;
        lower values flag a fixture where the models disagree.
        """
        lam_a, lam_b = self.dixon_coles.predict_lambdas(team_a, team_b, neutral=neutral)
        mc_p = _normalise(self.mc.run(lam_a, lam_b, stage=stage).outcome_proba())
        elo_p = self.elo.predict_proba(team_a, team_b, neutral=neutral)
        if stage in _KNOCKOUT_STAGES:
            half = elo_p["draw"] / 2.0
            elo_p = {"win_a": elo_p["win_a"] + half, "draw": 0.0,
                     "win_b": elo_p["win_b"] + half}
        tv = 0.5 * sum(abs(mc_p[k] - elo_p[k]) for k in ("win_a", "draw", "win_b"))
        return float(1.0 - tv)

    def _confidence(self, team_a: str, team_b: str) -> float:
        """Data-completeness proxy: are both teams present in the training set?"""
        seen = self.dixon_coles._index
        known = int(team_a in seen) + int(team_b in seen)
        return {0: 0.3, 1: 0.6, 2: 0.9}[known]
