"""Offline smoke tests for the Flask web UI (no network, injected engine)."""

from __future__ import annotations

import pytest

from football_predictor import webapp
from football_predictor.output.prediction_engine import PredictionEngine


@pytest.fixture
def client(sample_matches):
    # Inject a synthetic-fit engine so _get_engine() never touches the network.
    train = sample_matches[sample_matches["date"] < "2018-01-01"]
    webapp._engine = PredictionEngine(n_sims=5_000).fit(train)
    webapp._teams = sorted(set(sample_matches["team_a"]) | set(sample_matches["team_b"]))
    webapp.app.config.update(TESTING=True)
    return webapp.app.test_client()


def test_form_page_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Predictor de partidos" in r.get_data(as_text=True)


def test_prediction_page_renders(client):
    r = client.get("/?team_a=Brazil&team_b=Ghana&stage=group")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "Brazil vs Ghana" in html
    assert "Gana Brazil" in html and "Mercados" in html


def test_knockout_draw_bar_is_zero(client):
    # In a final the draw is folded into wins, so its bar renders at width 0%.
    r = client.get("/?team_a=Brazil&team_b=Ghana&stage=final")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "width: 0.0%" in html


def test_same_team_is_rejected(client):
    r = client.get("/?team_a=Brazil&team_b=Brazil")
    assert "distintos" in r.get_data(as_text=True)
