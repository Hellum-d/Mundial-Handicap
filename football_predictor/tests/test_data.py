"""Offline tests for real-dataset processing (no network access)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor.data.confederations import confederation_of
from football_predictor.data.flags import flag_code
from football_predictor.data.real_data import classify_tournament, process


def test_flag_code_mapping():
    assert flag_code("Brazil") == "br"
    assert flag_code("England") == "gb-eng"  # subdivision code
    assert flag_code("South Korea") == "kr"
    assert flag_code("United States") == "us"
    assert flag_code("Atlantis") is None  # unmapped -> no flag


def test_confederation_of_mapping():
    assert confederation_of("Brazil") == "CONMEBOL"
    assert confederation_of("Germany") == "UEFA"
    assert confederation_of("United States") == "CONCACAF"
    assert confederation_of("Senegal") == "CAF"
    assert confederation_of("Japan") == "AFC"
    assert confederation_of("New Zealand") == "OFC"
    # Unmapped teams fall back to OTHER.
    assert confederation_of("Atlantis") == "OTHER"


def test_classify_tournament_priorities():
    # Qualifiers are caught before continental finals.
    assert classify_tournament("UEFA Euro qualification") == ("qualifier", 0.4)
    assert classify_tournament("FIFA World Cup qualification") == ("qualifier", 0.4)
    # Finals.
    assert classify_tournament("FIFA World Cup") == ("world_cup", 0.9)
    assert classify_tournament("UEFA Euro") == ("continental", 0.7)
    assert classify_tournament("Copa América") == ("continental", 0.7)
    assert classify_tournament("UEFA Nations League") == ("nations_league", 0.5)
    assert classify_tournament("Friendly") == ("friendly", 0.2)
    # Unknown minor cups fall back.
    assert classify_tournament("Gulf Cup") == ("other", 0.3)


def _raw_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2022-12-18", "2018-06-14", "2026-06-11"],
            "home_team": ["Argentina", "Russia", "Mexico"],
            "away_team": ["France", "Saudi Arabia", "Poland"],
            "home_score": [3, 5, np.nan],  # last row = unplayed future fixture
            "away_score": [3, 0, np.nan],
            "tournament": ["FIFA World Cup", "FIFA World Cup", "FIFA World Cup"],
            "city": ["Lusail", "Moscow", "Guadalajara"],
            "country": ["Qatar", "Russia", "Mexico"],
            "neutral": [True, False, False],
        }
    )


def test_process_drops_unplayed_and_builds_schema():
    out = process(_raw_fixture())
    # The NaN-score future fixture is dropped.
    assert len(out) == 2
    # Exact canonical schema.
    assert list(out.columns) == [
        "date", "team_a", "team_b", "goals_a", "goals_b",
        "match_weight", "neutral", "competition",
    ]
    assert out["goals_a"].dtype == int and out["goals_b"].dtype == int
    assert out["neutral"].dtype == bool
    assert (out["competition"] == "world_cup").all()
    assert (out["match_weight"] == 0.9).all()


def test_process_sorted_by_date():
    out = process(_raw_fixture())
    assert out["date"].is_monotonic_increasing
