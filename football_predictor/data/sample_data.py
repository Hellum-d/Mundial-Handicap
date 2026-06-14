"""Deterministic synthetic match dataset.

Generates international fixtures from 2010-2022 by drawing goals from a
bivariate-Poisson process with fixed latent per-team attack/defence strengths
and a home advantage. World Cup years (2014/2018/2022) additionally get a
neutral-venue tournament among a subset of teams, which the back-tester uses as
its held-out folds.

The output is a tidy match-level frame with exactly the columns the models
consume: ``date, team_a, team_b, goals_a, goals_b, match_weight, neutral,
competition``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor import config

# 24 nations with a rough strength tier baked into the latent draws below.
TEAMS: list[str] = [
    "Brazil", "France", "Argentina", "Germany", "Spain", "England",
    "Belgium", "Portugal", "Netherlands", "Italy", "Croatia", "Uruguay",
    "Colombia", "Mexico", "Switzerland", "Denmark", "USA", "Senegal",
    "Japan", "Poland", "Serbia", "Morocco", "Ghana", "Ecuador",
]

# Competition -> (match_weight, probability of a neutral venue).
_COMPETITIONS = {
    "friendly": (0.2, 0.5),
    "qualifier": (0.4, 0.0),
    "continental": (0.7, 0.6),
}

_BASE_INTERCEPT = 0.15   # global log scoring rate
_HOME_ADV = 0.30         # latent home advantage (log scale)


def _latent_strengths(seed: int) -> dict[str, tuple[float, float]]:
    """Assign each team a fixed (attack, defence) strength on the log scale.

    Stronger teams (earlier in ``TEAMS``) get a small positive bias so the
    synthetic world has a realistic strength gradient rather than pure noise.
    """
    rng = np.random.default_rng(seed)
    strengths: dict[str, tuple[float, float]] = {}
    n = len(TEAMS)
    for i, team in enumerate(TEAMS):
        tier_bias = 0.80 * (1.0 - i / (n - 1))  # +0.80 (best) .. 0.0 (weakest)
        attack = tier_bias + rng.normal(0.0, 0.15)
        defence = tier_bias + rng.normal(0.0, 0.15)
        strengths[team] = (float(attack), float(defence))
    return strengths


def _sample_goals(
    rng: np.random.Generator,
    strengths: dict[str, tuple[float, float]],
    team_a: str,
    team_b: str,
    neutral: bool,
) -> tuple[int, int]:
    att_a, def_a = strengths[team_a]
    att_b, def_b = strengths[team_b]
    home = 0.0 if neutral else _HOME_ADV
    lam_a = np.exp(_BASE_INTERCEPT + att_a - def_b + home)
    lam_b = np.exp(_BASE_INTERCEPT + att_b - def_a)
    return int(rng.poisson(lam_a)), int(rng.poisson(lam_b))


def generate_sample_matches(seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Generate the full synthetic 2010-2022 dataset as a DataFrame."""
    strengths = _latent_strengths(seed)
    rng = np.random.default_rng(seed + 1)
    rows: list[dict] = []

    for year in range(2010, 2023):
        # --- regular international calendar -----------------------------
        n_regular = 120
        for _ in range(n_regular):
            a, b = rng.choice(TEAMS, size=2, replace=False)
            comp = rng.choice(list(_COMPETITIONS), p=[0.45, 0.35, 0.20])
            weight, neutral_p = _COMPETITIONS[comp]
            neutral = bool(rng.random() < neutral_p)
            ga, gb = _sample_goals(rng, strengths, a, b, neutral)
            month = int(rng.integers(1, 13))
            day = int(rng.integers(1, 28))
            rows.append(
                dict(
                    date=f"{year}-{month:02d}-{day:02d}",
                    team_a=a, team_b=b, goals_a=ga, goals_b=gb,
                    match_weight=weight, neutral=neutral, competition=comp,
                )
            )

        # --- World Cup (neutral) in 2014 / 2018 / 2022 ------------------
        if year in (2014, 2018, 2022):
            field = list(rng.choice(TEAMS, size=16, replace=False))
            for _ in range(64):
                a, b = rng.choice(field, size=2, replace=False)
                ga, gb = _sample_goals(rng, strengths, a, b, neutral=True)
                day = int(rng.integers(1, 28))
                rows.append(
                    dict(
                        date=f"{year}-06-{day:02d}",
                        team_a=a, team_b=b, goals_a=ga, goals_b=gb,
                        match_weight=0.9, neutral=True, competition="world_cup",
                    )
                )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_matches(regenerate: bool = False) -> pd.DataFrame:
    """Load the sample dataset, generating and caching it on first use.

    Args:
        regenerate: Force regeneration even if the cached CSV exists.
    """
    path = config.SAMPLE_MATCHES_CSV
    if path.exists() and not regenerate:
        return pd.read_csv(path, parse_dates=["date"])
    df = generate_sample_matches()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":  # pragma: no cover
    frame = load_matches(regenerate=True)
    print(f"Wrote {len(frame)} matches to {config.SAMPLE_MATCHES_CSV}")
    print(frame.head())
