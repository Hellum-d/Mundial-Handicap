"""Tests for the Elo and Dixon-Coles models and the prediction engine."""

from __future__ import annotations

import pytest

from football_predictor.models.dixon_coles import DixonColesModel
from football_predictor.models.elo_model import EloRatingSystem
from football_predictor.output.prediction_engine import PredictionEngine
from football_predictor.output.schemas import MatchPrediction


def _is_distribution(p: dict[str, float]) -> bool:
    return (
        all(0.0 <= v <= 1.0 for v in p.values())
        and abs(sum(p.values()) - 1.0) < 1e-6
    )


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        EloRatingSystem().predict_proba("Brazil", "Japan")
    with pytest.raises(RuntimeError):
        DixonColesModel().predict_lambdas("Brazil", "Japan")


def test_elo_outputs_valid_distribution(sample_matches):
    elo = EloRatingSystem().fit(sample_matches)
    assert _is_distribution(elo.predict_proba("Brazil", "Ghana", neutral=True))


def test_dixon_coles_outputs_valid_distribution(sample_matches):
    dc = DixonColesModel().fit(sample_matches)
    assert _is_distribution(dc.predict_proba("Brazil", "Ghana", neutral=True))


def test_stronger_team_favoured(sample_matches):
    """Brazil (top tier) should be favoured over Ghana (bottom tier)."""
    dc = DixonColesModel().fit(sample_matches)
    p = dc.predict_proba("Brazil", "Ghana", neutral=True)
    assert p["win_a"] > p["win_b"]

    elo = EloRatingSystem().fit(sample_matches)
    assert elo.expected_score("Brazil", "Ghana", neutral=True) > 0.5


def test_dixon_coles_lambdas_positive(sample_matches):
    dc = DixonColesModel().fit(sample_matches)
    lam_a, lam_b = dc.predict_lambdas("Brazil", "Ghana", neutral=True)
    assert lam_a > 0 and lam_b > 0
    assert lam_a > lam_b  # stronger attack, weaker opposing defence


def test_unknown_team_is_average(sample_matches):
    dc = DixonColesModel().fit(sample_matches)
    att, deff = dc._strength("Atlantis")
    assert att == 0.0 and deff == 0.0


def test_home_advantage_increases_lambda(sample_matches):
    dc = DixonColesModel().fit(sample_matches)
    home_lam, _ = dc.predict_lambdas("Brazil", "Ghana", neutral=False)
    neutral_lam, _ = dc.predict_lambdas("Brazil", "Ghana", neutral=True)
    assert home_lam > neutral_lam


def test_engine_predict_returns_valid_schema(sample_matches):
    engine = PredictionEngine(n_sims=20_000).fit(sample_matches)
    pred = engine.predict("Brazil", "Argentina", stage="group", neutral=True)
    assert isinstance(pred, MatchPrediction)
    total = (
        pred.win_probability_a + pred.draw_probability + pred.win_probability_b
    )
    assert abs(total - 1.0) < 1e-6
    assert len(pred.top_scorelines) <= 5


def test_knockout_has_no_draw(sample_matches):
    engine = PredictionEngine(n_sims=20_000).fit(sample_matches)
    pred = engine.predict("Brazil", "Argentina", stage="final", neutral=True)
    assert pred.draw_probability == 0.0
    assert abs(pred.win_probability_a + pred.win_probability_b - 1.0) < 1e-6
