"""Tests for joblib persistence of the fitted engine."""

from __future__ import annotations

import pytest

from football_predictor.output.prediction_engine import (
    PredictionEngine,
    load_or_train_engine,
)


def _train(sample_matches):
    train = sample_matches[sample_matches["date"] < "2018-01-01"]
    validation = sample_matches[
        (sample_matches["date"] >= "2018-01-01")
        & (sample_matches["date"] < "2018-06-01")
    ]
    return (
        PredictionEngine(n_sims=10_000)
        .fit(train)
        .fit_blend(validation)
        .fit_draw_calibration(validation)
    )


def test_save_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError):
        PredictionEngine().save(tmp_path / "e.joblib")


def test_save_load_preserves_fitted_parameters(tmp_path, sample_matches):
    eng = _train(sample_matches)
    path = eng.save(tmp_path / "engine.joblib")
    assert path.exists()

    loaded = PredictionEngine.load(path)
    assert loaded.is_fitted
    # Fitted scalars survive the round-trip exactly.
    assert loaded.w_mc == eng.w_mc
    assert loaded.w_elo == eng.w_elo
    assert loaded.draw_gamma == eng.draw_gamma
    # Base-model parameters are identical (deterministic analytic checks).
    assert loaded.elo.ratings == eng.elo.ratings
    assert loaded.dixon_coles.predict_lambdas(
        "Brazil", "Ghana", neutral=True
    ) == eng.dixon_coles.predict_lambdas("Brazil", "Ghana", neutral=True)
    assert loaded.dixon_coles.predict_proba(
        "Brazil", "Ghana", neutral=True
    ) == eng.dixon_coles.predict_proba("Brazil", "Ghana", neutral=True)


def test_loaded_engine_is_mc_deterministic(tmp_path, sample_matches):
    """Two loads from one file predict identically (RNG reset on load)."""
    path = _train(sample_matches).save(tmp_path / "e.joblib")
    a = PredictionEngine.load(path).predict_proba(
        "Brazil", "Argentina", stage="group", neutral=True
    )
    b = PredictionEngine.load(path).predict_proba(
        "Brazil", "Argentina", stage="group", neutral=True
    )
    assert a == b


def test_load_or_train_uses_cache(tmp_path, sample_matches):
    train = sample_matches[sample_matches["date"] < "2018-01-01"]
    path = tmp_path / "cache.joblib"
    assert not path.exists()
    e1 = load_or_train_engine(train, path=path)  # miss -> fits + caches
    assert path.exists()
    mtime = path.stat().st_mtime_ns
    e2 = load_or_train_engine(train, path=path)  # hit -> loads, no refit
    assert path.stat().st_mtime_ns == mtime  # cache untouched
    assert e1.w_mc == e2.w_mc
