# World Cup Match Prediction Engine

A modular probabilistic forecasting system for international football. This
repository contains a **runnable MVP vertical slice** of the full architecture
described below: it ingests a dataset, fits two complementary models, simulates
matches by Monte Carlo, emits a schema-validated prediction, and back-tests
itself against baselines with strict temporal splits.

```
data → DixonColes + Elo → MonteCarlo → blend → MatchPrediction (Pydantic)
                                  ↑
                            Backtester (temporal CV)
```

## What is implemented (MVP)

| Component | File | Notes |
|---|---|---|
| Sample data | `data/sample_data.py` | Deterministic synthetic 2010–2022 fixtures drawn from latent team strengths (no live feeds required) |
| Dixon-Coles bivariate Poisson | `models/dixon_coles.py` | Time-decayed, ridge-regularised MLE with the low-score `tau` correction; the scoreline engine |
| Dynamic Elo | `models/elo_model.py` | Walk-forward, importance-weighted K-factor, Clark margin multiplier; strong baseline / second opinion |
| Monte-Carlo engine | `simulation/monte_carlo.py` | 100k fully-vectorised Poisson sims; outcomes, scorelines, secondary markets, knockout AET + penalties |
| Prediction engine | `output/prediction_engine.py` | Blends the two model heads and assembles the output contract |
| Output schema | `output/schemas.py` | Pydantic `MatchPrediction`; enforces probabilities sum to 1 |
| Metrics | `evaluation/metrics.py` | Log loss (primary), Brier, Ranked Probability Score |
| Back-tester | `evaluation/backtester.py` | Temporal CV: train on the past, test on each held-out World Cup |

## Quick start

```bash
python -m football_predictor          # back-test + an example prediction
python -m football_predictor.data.sample_data   # (re)generate the sample CSV
python -m pytest football_predictor/tests -q     # 24 tests
```

Example back-test output (synthetic data; the ensemble beats the uniform and
Elo-only baselines on every fold):

```
=== Mean log loss across folds ===
  ensemble     0.9959
  elo_only     1.0166
  uniform      1.0986
```

## Design choices

- **Football as a low-scoring Poisson process.** No win/loss-ratio or
  deterministic "if A > B then A wins" logic — outcomes always come from
  sampling goals.
- **No future leakage.** Each back-test fold trains only on matches *before*
  the tested tournament; Elo is walked forward in date order.
- **Recency dominates.** Dixon-Coles applies an exponential time-decay
  (`config.DIXON_COLES_XI`) so distant matches fade.
- **Everything tunable lives in `config.py`.**

## Synthetic data

No external APIs are available in this environment, so `data/sample_data.py`
generates a deterministic dataset from fixed latent attack/defence strengths
sampled through the *same* bivariate-Poisson process the models assume. This
makes the back-test a genuine recovery test: a model that learns the latent
strengths should beat the naive baselines. Swapping in real feeds means
replacing this module with the ingestion clients below — the rest of the
pipeline is unchanged.

## Roadmap to the full system

The MVP intentionally implements 2 of the 7 planned models and a simple blend
in place of the stacked meta-learner. Still to come, in rough priority order:

1. **Real data ingestion** — StatsBomb / Understat / ClubElo / FIFA rankings
   clients + a `DataQualityPipeline` (schema validation, imputation, freshness).
2. **Feature store** (`features/`) — the ~85 engineered features (xG form,
   defensive/GK, player-level, contextual) backing the ML models.
3. **Remaining models** — XGBoost, LightGBM xG regressor, Bayesian hierarchical
   (PyMC), neural net (PyTorch), calibrated random forest.
4. **Stacked meta-learner** — non-negative logistic stack over out-of-fold
   predictions, replacing the current fixed Monte-Carlo/Elo blend.
5. **Calibration** — Platt/isotonic with an ECE < 0.05 holdout check.
6. **Tournament simulator** — full 32-team bracket, 10k tournament runs.
7. **Live adjustment + incremental learning** — injury/lineup/weather λ
   adjustments and post-match online updates.
8. **Explainability** — SHAP top-5 features on every prediction.

See the module docstrings for component-level detail.
