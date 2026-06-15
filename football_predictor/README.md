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
| Dixon-Coles bivariate Poisson | `models/dixon_coles.py` | Time-decayed, ridge-regularised MLE with the low-score `tau` correction; **confederation partial-pooling** of strengths; the scoreline engine. |
| Dynamic Elo | `models/elo_model.py` | Walk-forward, importance-weighted K-factor, Clark margin multiplier; strong baseline / second opinion. |
| Monte-Carlo engine | `simulation/monte_carlo.py` | 100k fully-vectorised Poisson sims; outcomes, scorelines, secondary markets, knockout AET + penalties. |
| Prediction engine | `output/prediction_engine.py` | Blends the two heads via a **1-parameter convex blend** (`w_mc + w_elo = 1`), fitted on a leak-free validation set (`fit_blend`); optional post-hoc draw calibration (`fit_draw_calibration`, off by default). joblib `save`/`load` so fitted engines persist and predictions are instant. |
| Predict CLI | `predict.py` | `python -m football_predictor.predict A B [--stage …] [--home]` — instant single-match prediction from the cached engine. |
| Market utilities | `market.py` | De-vig (proportional + Shin) and convex market-prior blend, for when an odds feed is available (none in the default dataset). |
| Output schema | `output/schemas.py` | Pydantic `MatchPrediction`; enforces probabilities sum to 1 and ranges. |
| Metrics | `evaluation/metrics.py` | Log loss (primary), Brier, RPS; bootstrap CIs and paired significance tests. |
| Back-tester | `evaluation/backtester.py` | Temporal CV with leak-free `train / validation / test` splits (`split_fold`), a 12-year training window, modern-era fold selection, and a pooled bootstrap `significance_report`. |

## Quick start

```bash
python -m football_predictor          # back-test on real data + an example prediction
python -m football_predictor.predict "Brazil" "Argentina" --stage final  # instant prediction
python -m football_predictor.data.real_data    # (re)download + cache the real dataset
python -m pytest football_predictor/tests -q   # 53 offline tests
```

The first run downloads the dataset (~3.7 MB, CC0) and caches it under `data/`,
and the first prediction fits and caches the engine (`data/engine.joblib`, via
joblib) — so subsequent predictions are effectively instant (no refit). Those
caches are git-ignored and regenerated on demand. The test suite is fully
offline (synthetic + in-memory fixtures).

Real-data back-test (held-out World Cups 2010–2022, 64 matches each, 12y
training window):

```
=== Mean log loss across folds ===
  ensemble     1.0038
  elo_only     1.0215
  uniform      1.0986

=== Pooled log loss, 90% bootstrap CI (n=256 matches) ===
  ensemble     1.0038  [0.9627, 1.0458]
  elo_only     1.0215  [0.9823, 1.0617]
  uniform      1.0986  [1.0986, 1.0986]
=== ensemble - baseline (negative = ensemble better) ===
  ensemble - elo_only  -0.0177  [-0.0409, +0.0058]  (n.s.)
  ensemble - uniform   -0.0948  [-0.1360, -0.0528]  (significant)
```

Read this honestly: the ensemble **significantly** beats the uniform baseline,
but its edge over Elo-only — real as a point estimate (−0.018) — is **not
statistically significant** on 256 matches (the difference CI straddles 0).
With three or four World Cups of test data, that is the expected ceiling on
what can be claimed; it is exactly the small-sample caveat to keep in mind, not
a number to oversell.

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
- **Partial pooling, not isolated strengths.** Dixon-Coles shrinks each team
  toward its confederation mean so sparsely-observed sides inherit a sensible
  baseline.
- **Improvements are measured, and kept only if they earn it.** Each accuracy
  lever was back-tested: confederation pooling is ~neutral on the WC metric and
  left on for robustness; post-hoc draw calibration *worsened* held-out log loss
  (validation-vs-tournament draw-rate mismatch) and is therefore off by default;
  bootstrap CIs exist precisely so these calls are made on evidence, not vibes.
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

Done: real-data ingestion (results), Dixon-Coles (with confederation pooling) +
Elo, Monte Carlo, the leak-free back-tester with bootstrap significance, the
fitted convex blend, and the de-vig/market-blend utilities. Still ahead, in
rough priority order:

1. **Market prior** — de-vig + blend utilities exist (`market.py`: proportional
   and Shin de-vigging, convex market blend); still needs a live odds feed
   wired in (the results-only default dataset has none). Highest accuracy ROI.
2. **Richer ingestion + feature store** — StatsBomb/Elo/ratings feeds and the
   engineered features (xG form, defensive/GK, player-level, contextual),
   behind a `DataQualityPipeline`.
3. **Remaining models** — XGBoost, LightGBM xG regressor, Bayesian hierarchical
   (PyMC), neural net (PyTorch), calibrated random forest.
4. **Stacked meta-learner** — replacing the 1-parameter blend once there are
   enough models and out-of-fold predictions to justify it.
5. **Post-hoc calibration** — a draw-probability multiplier (`fit_draw_calibration`)
   already exists but is off by default (it needs a World-Cup-like validation
   set; the pre-tournament slice mismatches). Extend with an ECE check and a
   better-matched calibration set.
6. **Tournament simulator** — full 32-team bracket, 10k tournament runs.
7. **Live adjustment + incremental learning** — injury/lineup/weather λ
   adjustments and post-match online updates.
8. **Explainability** — SHAP top-5 features on every prediction.

See the module docstrings for component-level detail.
