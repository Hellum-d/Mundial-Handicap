# World Cup Match Prediction Engine

A modular probabilistic forecasting system for international football. This
repository is a **runnable MVP vertical slice** of the larger architecture: it
ingests real match results, fits two complementary models, simulates matches by
Monte Carlo, blends them into a schema-validated prediction, and back-tests
itself against baselines with strict, leak-free temporal splits.

```
data → DixonColes + Elo → MonteCarlo → convex blend → MatchPrediction (Pydantic)
                                  ↑                          ↑
                            Backtester (temporal CV, leak-free train/val/test)
```

## Objectives & non-goals

**Primary objective.** Emit *well-calibrated* outcome probabilities that beat
the naive baselines — a uniform `(1/3, 1/3, 1/3)` forecast and a dynamic
Elo-only model — on held-out World Cups, measured by log loss (primary), Brier,
and RPS. This is met: on the real held-out World Cups (2010–2022) the ensemble
averages **~1.00** log loss vs **~1.02** Elo-only and **~1.10** uniform.

**Explicit non-goal.** The original spec's target — *"outperform bookmaker
implied probabilities by at least 3% on log loss"* — is **rejected as
unrealistic** and is not a success criterion here. Closing odds are
near-efficient; beating them by ~3% (≈0.95 → 0.92) is an edge professionals
rarely sustain. The realistic bar is to *approach* the market and, on the
roadmap, to fold **de-vigged implied probabilities in as a feature/prior**
rather than to try to beat the close. Success = calibration + beating the
uniform/Elo baselines, not beating the market.

## What is implemented (MVP)

| Component | File | Notes |
|---|---|---|
| Real dataset | `data/real_data.py` | martj42 / Kaggle "International results" (CC0). Downloaded from the public GitHub mirror on first run, mapped to the canonical schema, `match_weight` derived from the tournament. **Default data source.** |
| Synthetic dataset | `data/sample_data.py` | Deterministic fixtures from latent team strengths; used by the offline test suite (no network). |
| Dixon-Coles bivariate Poisson | `models/dixon_coles.py` | Time-decayed, ridge-regularised MLE with the low-score `tau` correction; the scoreline engine. |
| Dynamic Elo | `models/elo_model.py` | Walk-forward, importance-weighted K-factor, Clark margin multiplier; strong baseline / second opinion. |
| Monte-Carlo engine | `simulation/monte_carlo.py` | 100k fully-vectorised Poisson sims; outcomes, scorelines, secondary markets, knockout AET + penalties. |
| Prediction engine | `output/prediction_engine.py` | Blends the two heads via a **1-parameter convex blend** (`w_mc + w_elo = 1`), default or fitted on a leak-free validation set (`fit_blend`). |
| Output schema | `output/schemas.py` | Pydantic `MatchPrediction`; enforces probabilities sum to 1 and ranges. |
| Metrics | `evaluation/metrics.py` | Log loss (primary), Brier, Ranked Probability Score. |
| Back-tester | `evaluation/backtester.py` | Temporal CV with leak-free `train / validation / test` splits (`split_fold`), a 12-year training window, and modern-era fold selection. |

## Quick start

```bash
python -m football_predictor          # back-test on real data + an example prediction
python -m football_predictor.data.real_data    # (re)download + cache the real dataset
python -m pytest football_predictor/tests -q   # 31 offline tests
```

The first run downloads the dataset (~3.7 MB, CC0) and caches it under
`data/`; those caches are git-ignored and regenerated on demand. The test suite
is fully offline (synthetic + in-memory fixtures).

Real-data back-test (held-out World Cups, train window 12y):

```
[world_cup 2010]  (64 matches, blend w_mc=0.73)   ensemble 0.967  elo 1.006  uniform 1.099
[world_cup 2014]  (64 matches, blend w_mc=0.70)   ensemble 0.989  elo 1.029  uniform 1.099
[world_cup 2018]  (64 matches, blend w_mc=0.74)   ensemble 0.990  elo 1.022  uniform 1.099
[world_cup 2022]  (64 matches, blend w_mc=0.67)   ensemble 1.069  elo 1.030  uniform 1.099

=== Mean log loss across folds ===
  ensemble     1.0039
  elo_only     1.0215
  uniform      1.0986
```

(The ensemble beats both baselines on the mean and on 2010/14/18; 2022 — a
high-upset World Cup — edges Elo-only, which is reported honestly rather than
hidden.)

## Design choices

- **Football as a low-scoring Poisson process.** No win/loss-ratio or
  deterministic "if A > B then A wins" logic — outcomes always come from
  sampling goals.
- **No future leakage.** Every fold splits strictly by date into
  `train < validation < test`; base models fit on `train`, the blend weight is
  fitted on the disjoint `validation`, models refit on all pre-test data, and
  the test World Cup is never seen during any fitting. Elo is walked forward in
  date order.
- **Recency dominates.** Dixon-Coles applies an exponential time-decay
  (`config.DIXON_COLES_XI`, ~1-year half-life); a 12-year training window drops
  data that is both negligible under the decay and expensive to fit.
- **A 1-parameter blend, not a stack.** With two models and ~64-match test eras
  a convex simplex blend (always a valid distribution, one degree of freedom)
  is preferred to logistic-L2 stacking, which would over-parameterise.
- **Everything tunable lives in `config.py`.**

## Data

The default source is the `martj42/international_results` dataset (the upstream
of the Kaggle "International football results 1872-2024" set), **CC0-1.0**
(public domain), fetched from the public GitHub raw mirror — no Kaggle login,
API token, or scraping. It is *results-only* (no xG/lineups), which is exactly
what the two MVP models consume; the columns map cleanly to the canonical
schema and `match_weight` is derived from the tournament name. The synthetic
generator is retained purely so the test suite runs offline and deterministically.

## Roadmap to the full system

Done: real-data ingestion (results), Dixon-Coles + Elo, Monte Carlo, the
leak-free back-tester, and the fitted convex blend. Still ahead, in rough
priority order:

1. **Market prior** — de-vigged bookmaker implied probabilities as a
   feature/prior (highest expected accuracy ROI).
2. **Richer ingestion + feature store** — StatsBomb/Elo/ratings feeds and the
   engineered features (xG form, defensive/GK, player-level, contextual),
   behind a `DataQualityPipeline`.
3. **Remaining models** — XGBoost, LightGBM xG regressor, Bayesian hierarchical
   (PyMC), neural net (PyTorch), calibrated random forest.
4. **Stacked meta-learner** — replacing the 1-parameter blend once there are
   enough models and out-of-fold predictions to justify it.
5. **Post-hoc calibration** — e.g. a draw-probability temperature fitted on the
   leak-free validation slice, with an ECE check.
6. **Tournament simulator** — full 32-team bracket, 10k tournament runs.
7. **Live adjustment + incremental learning** — injury/lineup/weather λ
   adjustments and post-match online updates.
8. **Explainability** — SHAP top-5 features on every prediction.

See the module docstrings for component-level detail.
