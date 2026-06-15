"""Real international match results.

Source: the ``martj42/international_results`` dataset (the upstream of the
Kaggle "International football results 1872-2024" set), licensed **CC0-1.0**
(public domain). It is fetched from the public GitHub raw mirror, so no Kaggle
login, API token, or scraping is involved.

The raw file is *results-only* (no xG/lineups) with columns
``date, home_team, away_team, home_score, away_score, tournament, city,
country, neutral``. This module maps it onto the canonical schema the pipeline
consumes and derives the one column the raw data lacks, ``match_weight``, from
the ``tournament`` name.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from football_predictor import config

# Continental *final* tournaments (lowercase substrings). Qualifiers are caught
# earlier, so these only match the finals themselves.
_CONTINENTAL_FINALS: tuple[str, ...] = (
    "uefa euro",
    "copa am",  # Copa America (accent-insensitive)
    "african cup of nations",
    "afc asian cup",
    "gold cup",
    "concacaf championship",
    "confederations cup",
    "ofc nations cup",
    "oceania nations cup",
)


def classify_tournament(tournament: str) -> tuple[str, float]:
    """Map a raw ``tournament`` name to ``(competition_category, match_weight)``.

    Rules are checked in priority order; the first match wins. Qualifiers are
    tested before continental finals so that e.g. "UEFA Euro qualification"
    classifies as a qualifier, not a continental final.
    """
    tl = str(tournament).lower()
    if "qualif" in tl:
        return "qualifier", 0.4
    if "fifa world cup" in tl:
        return "world_cup", 0.9
    if "nations league" in tl:
        return "nations_league", 0.5
    if tl == "friendly":
        return "friendly", 0.2
    if any(key in tl for key in _CONTINENTAL_FINALS):
        return "continental", 0.7
    return "other", 0.3


def download_raw(force: bool = False) -> Path:
    """Download the raw results CSV to the cache, returning its path.

    Args:
        force: Re-download even if the cached raw file already exists.
    """
    dest = config.RAW_RESULTS_CSV
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(config.REAL_RESULTS_URL, dest)
    except Exception as exc:  # network/offline
        raise RuntimeError(
            f"Could not download the results dataset from {config.REAL_RESULTS_URL!r}. "
            f"Download it manually and place it at {dest}."
        ) from exc
    return dest


def process(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw results frame into the canonical pipeline schema.

    Drops unplayed (NaN-score) future fixtures, coerces types, and derives
    ``competition`` and ``match_weight`` from the tournament name.
    """
    df = raw.dropna(subset=["home_score", "away_score"]).copy()
    classified = df["tournament"].map(classify_tournament)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"]),
            "team_a": df["home_team"].astype(str),
            "team_b": df["away_team"].astype(str),
            "goals_a": df["home_score"].astype(int),
            "goals_b": df["away_score"].astype(int),
            "match_weight": classified.map(lambda x: x[1]).to_numpy(),
            "neutral": df["neutral"].astype(bool),
            "competition": classified.map(lambda x: x[0]).to_numpy(),
        }
    )
    return out.sort_values("date").reset_index(drop=True)


def load_real_matches(regenerate: bool = False) -> pd.DataFrame:
    """Load the processed real dataset, downloading and caching on first use.

    Args:
        regenerate: Force a re-download and re-process even if caches exist.
    """
    cache = config.REAL_MATCHES_CSV
    if cache.exists() and not regenerate:
        return pd.read_csv(cache, parse_dates=["date"])

    raw_path = download_raw(force=regenerate)
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    out = process(raw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


if __name__ == "__main__":  # pragma: no cover
    frame = load_real_matches(regenerate=True)
    print(f"Wrote {len(frame)} matches to {config.REAL_MATCHES_CSV}")
    print(frame["competition"].value_counts().to_string())
    print(frame.tail())
