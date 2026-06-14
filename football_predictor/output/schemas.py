"""Pydantic output schema for a single match prediction.

Every prediction the engine emits is validated through :class:`MatchPrediction`
before it leaves the system. The schema enforces the structural invariants that
matter downstream: probabilities live in ``[0, 1]`` and the three outcome
probabilities sum to 1. This is the MVP subset of the full output contract
described in the project README (no SHAP/live-event fields yet).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Stage = Literal["group", "r16", "qf", "sf", "third_place", "final"]


class MatchPrediction(BaseModel):
    """Validated probabilistic forecast for one fixture."""

    # Identity
    team_a: str
    team_b: str
    stage: Stage = "group"
    neutral: bool = True
    prediction_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Outcome probabilities (sum to 1.0; draw is 0.0 in knockout stages)
    win_probability_a: float = Field(ge=0.0, le=1.0)
    draw_probability: float = Field(ge=0.0, le=1.0)
    win_probability_b: float = Field(ge=0.0, le=1.0)

    # Expected goals with 90% confidence intervals
    xg_a: float = Field(ge=0.0)
    xg_b: float = Field(ge=0.0)
    xg_a_ci_90: tuple[float, float]
    xg_b_ci_90: tuple[float, float]

    # Scorelines (top 5, descending probability)
    top_scorelines: list[dict]

    # Secondary markets
    over_2_5_probability: float = Field(ge=0.0, le=1.0)
    btts_probability: float = Field(ge=0.0, le=1.0)
    p_team_a_scores: float = Field(ge=0.0, le=1.0)
    p_team_b_scores: float = Field(ge=0.0, le=1.0)
    clean_sheet_a: float = Field(ge=0.0, le=1.0)
    clean_sheet_b: float = Field(ge=0.0, le=1.0)

    # Model quality
    confidence_score: float = Field(ge=0.0, le=1.0)
    model_agreement_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_probabilities_sum_to_one(self) -> "MatchPrediction":
        total = (
            self.win_probability_a + self.draw_probability + self.win_probability_b
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Outcome probabilities must sum to 1.0, got {total:.6f}")
        return self
