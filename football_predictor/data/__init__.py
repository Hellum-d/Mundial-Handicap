"""Data layer: synthetic sample dataset generation and loading.

The full system ingests StatsBomb / Understat / Elo / Transfermarkt feeds (see
the README). For a self-contained, dependency-free MVP this package instead
generates a deterministic *synthetic* dataset from latent per-team attack and
defence strengths, sampled through the same bivariate-Poisson process the
models assume. That makes the back-test meaningful: a model that recovers the
latent strengths should beat the naive baselines.
"""

from football_predictor.data.sample_data import (
    generate_sample_matches,
    load_matches,
)
from football_predictor.data.real_data import load_real_matches

__all__ = ["generate_sample_matches", "load_matches", "load_real_matches"]
