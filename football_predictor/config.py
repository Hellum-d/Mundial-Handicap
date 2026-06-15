"""Global configuration constants for the prediction engine.

Kept deliberately small and explicit. Anything that is a tunable
hyperparameter (and not a hard structural choice) lives here so it can be
swept later without hunting through the codebase.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = PACKAGE_ROOT.parent
DATA_DIR: Path = PACKAGE_ROOT / "data"
SAMPLE_MATCHES_CSV: Path = DATA_DIR / "sample_matches.csv"

# --- Real dataset (martj42 / Kaggle "International results", CC0) -----------
# Public mirror, no login or scraping required.
REAL_RESULTS_URL: str = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
RAW_RESULTS_CSV: Path = DATA_DIR / "results_raw.csv"      # downloaded, untouched
REAL_MATCHES_CSV: Path = DATA_DIR / "real_matches.csv"    # processed, canonical
MODEL_CACHE: Path = DATA_DIR / "engine.joblib"            # fitted engine cache
# Only fit on the most recent N years before a fixture: with DIXON_COLES_XI the
# weight of a 12-year-old match is ~2.5e-4, so older data is both negligible and
# expensive (the full 49k-row history makes each Dixon-Coles fit ~65s).
TRAIN_WINDOW_YEARS: int = 12
# Length of the leak-free validation slice carved out immediately before each
# test tournament (used to fit calibration / blend weights, never the test set).
BACKTEST_VALIDATION_MONTHS: int = 12

# --- Dixon-Coles -----------------------------------------------------------
# Time-decay rate (per day) for match weighting in the MLE fit.
# xi ~= 0.0019 => a match ~1 year old keeps ~50% weight.
DIXON_COLES_XI: float = 0.0019
# Ridge penalty on attack/defence strengths (breaks the additive degeneracy
# between the global intercept and the per-team strengths, and stabilises
# teams with few matches).
DIXON_COLES_RIDGE: float = 1e-3
# With confederation pooling on, DIXON_COLES_RIDGE shrinks each team toward its
# confederation mean; this weak global ridge additionally anchors the whole set
# (keeps confederation means from drifting and stabilises tiny confederations).
DIXON_COLES_RIDGE_GLOBAL: float = 1e-4
# Maximum goals modelled in the analytic scoreline grid.
MAX_GOALS_GRID: int = 10

# --- Elo -------------------------------------------------------------------
ELO_INITIAL: float = 1500.0
ELO_HOME_ADVANTAGE: float = 65.0  # rating points added to the home side
# K-factor by match importance. Friendlies move ratings little; WC finals a lot.
ELO_K_BY_WEIGHT: dict[float, float] = {
    1.0: 60.0,   # WC final / third place
    0.95: 55.0,  # WC knockout
    0.9: 50.0,   # WC group
    0.7: 35.0,   # continental
    0.4: 25.0,   # qualifier
    0.2: 10.0,   # friendly
}

# --- Monte Carlo -----------------------------------------------------------
MC_DEFAULT_SIMS: int = 100_000
# Extra time is roughly a third of a match.
EXTRA_TIME_FRACTION: float = 0.33

# --- Ensemble blend (stand-in for the full meta-learner) -------------------
# Final outcome probabilities = blend of the Monte-Carlo (Dixon-Coles driven)
# distribution and the Elo model's own outcome probabilities.
BLEND_WEIGHT_MONTE_CARLO: float = 0.65
BLEND_WEIGHT_ELO: float = 0.35

# --- Reproducibility -------------------------------------------------------
RANDOM_SEED: int = 42
