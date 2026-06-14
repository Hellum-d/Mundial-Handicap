"""Abstract base class shared by every predictive model.

The full system (see ``README.md``) targets seven models plus a meta-learner.
The MVP implements two — :class:`~football_predictor.models.elo_model.EloRatingSystem`
and :class:`~football_predictor.models.dixon_coles.DixonColesModel` — but they
share this interface so the ensemble and back-tester can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

# A match outcome is reported as a probability over three classes.
OutcomeProba = dict[str, float]  # keys: "win_a", "draw", "win_b"


class BaseModel(ABC):
    """Common interface for outcome-probability models.

    Concrete models must implement :meth:`fit` and :meth:`predict_proba`.
    Probabilities returned by :meth:`predict_proba` must be non-negative and
    sum to 1 (within floating-point tolerance).
    """

    #: Set to ``True`` by :meth:`fit`. Guards against predicting on an unfit model.
    is_fitted: bool = False

    @abstractmethod
    def fit(self, matches: pd.DataFrame) -> "BaseModel":
        """Fit the model on a match-level dataframe.

        Args:
            matches: One row per historical match. Required columns are
                ``date, team_a, team_b, goals_a, goals_b, match_weight,
                neutral``. Implementations may use a subset.

        Returns:
            ``self``, to allow fluent ``model.fit(df).predict_proba(...)``.
        """

    @abstractmethod
    def predict_proba(
        self, team_a: str, team_b: str, *, neutral: bool = True
    ) -> OutcomeProba:
        """Return ``{"win_a", "draw", "win_b"}`` probabilities for a fixture.

        Args:
            team_a: Home (or first) team name.
            team_b: Away (or second) team name.
            neutral: Whether the venue is neutral (the World Cup default).
        """

    def _check_fitted(self) -> None:
        """Raise if :meth:`predict_proba` is called before :meth:`fit`."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling predict_proba()."
            )
