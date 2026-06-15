"""Dixon-Coles bivariate Poisson model.

Implements the classic Dixon & Coles (1997) model for association-football
scores. Each team carries an attack strength and a defence strength; the two
scoring rates for a fixture are

    lambda_A = exp(c + attack_A - defence_B + gamma * is_home_A)
    lambda_B = exp(c + attack_B - defence_A)

Goals are *almost* two independent Poissons, except that low-score outcomes
(0-0, 1-0, 0-1, 1-1) are correlated; the Dixon-Coles ``tau`` term corrects the
joint probability of exactly those four cells via a single dependence
parameter ``rho``.

Parameters are fit by maximum likelihood with an exponential time-decay on the
match weights (recent matches dominate) and a small ridge penalty on the
attack/defence vectors. The ridge breaks the additive degeneracy between the
intercept and the per-team strengths and shrinks teams with few matches toward
the average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from football_predictor import config
from football_predictor.data.confederations import confederation_of
from football_predictor.models.base_model import BaseModel, OutcomeProba

# Numerical floor so log(tau) and log(lambda) stay finite.
_EPS = 1e-10


def _tau(
    ga: np.ndarray, gb: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> np.ndarray:
    """Dixon-Coles low-score correction, vectorised over matches.

    Returns a multiplier of 1.0 everywhere except the four correlated cells.
    """
    tau = np.ones_like(lam, dtype=float)
    m00 = (ga == 0) & (gb == 0)
    m01 = (ga == 0) & (gb == 1)
    m10 = (ga == 1) & (gb == 0)
    m11 = (ga == 1) & (gb == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


class DixonColesModel(BaseModel):
    """Time-weighted, ridge-regularised Dixon-Coles bivariate Poisson."""

    def __init__(
        self,
        xi: float = config.DIXON_COLES_XI,
        ridge: float = config.DIXON_COLES_RIDGE,
        max_goals: int = config.MAX_GOALS_GRID,
        confederation_pooling: bool = True,
        ridge_global: float = config.DIXON_COLES_RIDGE_GLOBAL,
    ) -> None:
        """Initialise an unfitted model.

        Args:
            xi: Per-day time-decay rate; match weight is ``exp(-xi * age_days)``.
            ridge: L2 penalty on the attack/defence strengths. With
                ``confederation_pooling`` this shrinks each team toward its
                confederation mean; otherwise toward zero (the global mean).
            max_goals: Largest goal count modelled in the analytic score grid.
            confederation_pooling: If True, partial-pool strengths toward the
                team's confederation mean (helps sparsely-observed teams).
            ridge_global: Weak L2 anchor toward zero applied alongside the
                pooling penalty to keep confederation means identifiable.
        """
        self.xi = xi
        self.ridge = ridge
        self.max_goals = max_goals
        self.confederation_pooling = confederation_pooling
        self.ridge_global = ridge_global

        self.teams: list[str] = []
        self._index: dict[str, int] = {}
        self.attack: np.ndarray = np.empty(0)
        self.defence: np.ndarray = np.empty(0)
        self.intercept: float = 0.0
        self.home_adv: float = 0.0
        self.rho: float = 0.0

    # -- parameter packing -------------------------------------------------
    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        n = len(self.teams)
        attack = theta[:n]
        defence = theta[n : 2 * n]
        intercept = theta[2 * n]
        home_adv = theta[2 * n + 1]
        rho = theta[2 * n + 2]
        return attack, defence, intercept, home_adv, rho

    # -- fitting -----------------------------------------------------------
    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        """Fit attack/defence/home/rho by penalised, time-weighted MLE."""
        df = matches.dropna(subset=["team_a", "team_b", "goals_a", "goals_b"]).copy()

        self.teams = sorted(set(df["team_a"]) | set(df["team_b"]))
        self._index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        a = df["team_a"].map(self._index).to_numpy()
        b = df["team_b"].map(self._index).to_numpy()
        ga = df["goals_a"].to_numpy(dtype=int)
        gb = df["goals_b"].to_numpy(dtype=int)
        is_home = (~df["neutral"].astype(bool)).to_numpy(dtype=float)

        # Exponential time decay relative to the most recent match.
        dates = pd.to_datetime(df["date"])
        age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
        decay = np.exp(-self.xi * age_days)
        # Fold in the explicit competition importance weight too.
        weights = decay * df["match_weight"].to_numpy(dtype=float)

        # Confederation groups for hierarchical (partial-pooling) shrinkage.
        if self.confederation_pooling:
            confs = [confederation_of(t) for t in self.teams]
            conf_index = {c: i for i, c in enumerate(sorted(set(confs)))}
            conf_ids = np.array([conf_index[c] for c in confs])
            conf_counts = np.bincount(conf_ids).astype(float)

        def _penalty(attack: np.ndarray, defence: np.ndarray) -> float:
            if not self.confederation_pooling:
                return self.ridge * (np.sum(attack**2) + np.sum(defence**2))
            # Shrink toward the confederation mean, plus a weak global anchor.
            att_mean = np.bincount(conf_ids, weights=attack) / conf_counts
            def_mean = np.bincount(conf_ids, weights=defence) / conf_counts
            att_dev = attack - att_mean[conf_ids]
            def_dev = defence - def_mean[conf_ids]
            return self.ridge * (np.sum(att_dev**2) + np.sum(def_dev**2)) + (
                self.ridge_global * (np.sum(attack**2) + np.sum(defence**2))
            )

        def neg_log_lik(theta: np.ndarray) -> float:
            attack, defence, intercept, home_adv, rho = self._unpack(theta)
            log_lam = intercept + attack[a] - defence[b] + home_adv * is_home
            log_mu = intercept + attack[b] - defence[a]
            lam = np.exp(np.clip(log_lam, -10, 10))
            mu = np.exp(np.clip(log_mu, -10, 10))

            base = ga * log_lam - lam + gb * log_mu - mu
            tau = np.clip(_tau(ga, gb, lam, mu, rho), _EPS, None)
            ll = weights * (base + np.log(tau))

            return -float(np.sum(ll)) + _penalty(attack, defence)

        # Initial guess: zero strengths, league-average scoring intercept.
        mean_goals = float(np.mean(np.concatenate([ga, gb])))
        theta0 = np.zeros(2 * n + 3)
        theta0[2 * n] = np.log(max(mean_goals, 0.1))  # intercept
        theta0[2 * n + 1] = 0.25  # home advantage
        theta0[2 * n + 2] = -0.05  # rho

        bounds = [(-3.0, 3.0)] * (2 * n)  # attack / defence
        bounds += [(-2.0, 2.0), (-1.0, 1.0), (-0.2, 0.2)]  # intercept, home, rho

        result = minimize(
            neg_log_lik,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500},
        )

        attack, defence, intercept, home_adv, rho = self._unpack(result.x)
        # Centre strengths for interpretability (does not change predictions
        # because the intercept absorbs the shift).
        self.attack = attack - attack.mean()
        self.defence = defence - defence.mean()
        self.intercept = float(intercept + attack.mean() - defence.mean())
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.is_fitted = True
        return self

    # -- prediction --------------------------------------------------------
    def _strength(self, team: str) -> tuple[float, float]:
        """Return (attack, defence) for a team; zeros for an unseen team."""
        i = self._index.get(team)
        if i is None:
            return 0.0, 0.0
        return float(self.attack[i]), float(self.defence[i])

    def predict_lambdas(
        self, team_a: str, team_b: str, *, neutral: bool = True
    ) -> tuple[float, float]:
        """Expected goals ``(lambda_A, lambda_B)`` for a fixture."""
        self._check_fitted()
        att_a, def_a = self._strength(team_a)
        att_b, def_b = self._strength(team_b)
        home = 0.0 if neutral else 1.0
        lam = np.exp(self.intercept + att_a - def_b + self.home_adv * home)
        mu = np.exp(self.intercept + att_b - def_a)
        return float(lam), float(mu)

    def score_matrix(
        self, team_a: str, team_b: str, *, neutral: bool = True
    ) -> np.ndarray:
        """Analytic joint scoreline probabilities ``P[i, j]`` for i,j goals.

        Includes the Dixon-Coles low-score correction and is renormalised so
        the truncated grid sums to 1.
        """
        lam, mu = self.predict_lambdas(team_a, team_b, neutral=neutral)
        n = self.max_goals + 1
        i = np.arange(n)
        # Independent Poisson pmfs via the log-gamma factorial.
        log_fact = np.array([np.sum(np.log(np.arange(1, k + 1))) for k in i])
        pa = np.exp(i * np.log(lam) - lam - log_fact)
        pb = np.exp(i * np.log(mu) - mu - log_fact)
        grid = np.outer(pa, pb)

        # Apply tau to the four correlated cells.
        grid[0, 0] *= 1.0 - lam * mu * self.rho
        grid[0, 1] *= 1.0 + lam * self.rho
        grid[1, 0] *= 1.0 + mu * self.rho
        grid[1, 1] *= 1.0 - self.rho

        grid = np.clip(grid, 0.0, None)
        return grid / grid.sum()

    def predict_proba(
        self, team_a: str, team_b: str, *, neutral: bool = True
    ) -> OutcomeProba:
        """Win/draw/win probabilities from the analytic scoreline grid."""
        grid = self.score_matrix(team_a, team_b, neutral=neutral)
        win_a = float(np.tril(grid, -1).sum())  # rows > cols  (A scores more)
        draw = float(np.trace(grid))
        win_b = float(np.triu(grid, 1).sum())
        total = win_a + draw + win_b
        return {"win_a": win_a / total, "draw": draw / total, "win_b": win_b / total}
