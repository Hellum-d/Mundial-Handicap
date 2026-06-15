"""Minimal Flask web UI for single-match predictions.

Run it with::

    python -m football_predictor.webapp

then open http://127.0.0.1:5000 in a browser. Type two teams, pick a stage, and
the page shows the win/draw/win probabilities, expected goals, most likely
scorelines, and secondary markets as plain CSS bar charts (no external
CDN — works fully offline).

The fitted engine is cached (joblib), so after the first request predictions
are instant.
"""

from __future__ import annotations

import pandas as pd
from flask import Flask, render_template_string, request

from football_predictor import config
from football_predictor.data.real_data import load_real_matches
from football_predictor.output.prediction_engine import load_or_train_engine

app = Flask(__name__)

_STAGES = [
    ("group", "Fase de grupos"),
    ("r16", "Octavos"),
    ("qf", "Cuartos"),
    ("sf", "Semifinal"),
    ("third_place", "Tercer puesto"),
    ("final", "Final"),
]

# Lazily-initialised, then cached for the process lifetime.
_engine = None
_teams: list[str] = []


def _get_engine():
    """Load (or fit+cache) the engine and the team list on first use."""
    global _engine, _teams
    if _engine is None:
        matches = load_real_matches()
        cutoff = matches["date"].max() - pd.DateOffset(years=config.TRAIN_WINDOW_YEARS)
        _engine = load_or_train_engine(matches[matches["date"] >= cutoff])
        _teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
    return _engine, _teams


_PAGE = """
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictor de partidos</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --ink:#e2e8f0; --muted:#94a3b8;
          --a:#2563eb; --d:#9ca3af; --b:#dc2626; --acc:#22c55e; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .wrap { max-width: 720px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 1.4rem; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 24px; font-size: .9rem; }
  .card { background: var(--card); border-radius: 14px; padding: 20px;
          margin-bottom: 20px; box-shadow: 0 1px 0 rgba(255,255,255,.04) inset; }
  form { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  label { font-size: .78rem; color: var(--muted); display:block; margin-bottom:4px; }
  input[type=text], select { width:100%; padding:10px; border-radius:9px;
      border:1px solid #334155; background:#0b1220; color:var(--ink); font-size:1rem; }
  .row-full { grid-column: 1 / -1; display:flex; gap:14px; align-items:center;
              flex-wrap: wrap; }
  button { background: var(--acc); color:#06240f; border:0; padding:11px 20px;
           border-radius:9px; font-weight:700; font-size:1rem; cursor:pointer; }
  .chk { display:flex; align-items:center; gap:8px; color:var(--muted);
         font-size:.9rem; }
  .matchup { font-size:1.15rem; font-weight:700; margin: 0 0 16px; }
  .bar { margin: 10px 0; }
  .bar .top { display:flex; justify-content:space-between; font-size:.9rem;
              margin-bottom:5px; }
  .bar .track { background:#0b1220; border-radius:7px; height:22px; overflow:hidden; }
  .bar .fill { height:100%; border-radius:7px; min-width: 2px;
               transition: width .3s; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:8px 24px; }
  .sl { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:.9rem; }
  .sl .score { width:46px; font-variant-numeric: tabular-nums; color:var(--muted); }
  .sl .track { flex:1; background:#0b1220; border-radius:6px; height:16px; }
  .sl .fill { height:100%; background:#475569; border-radius:6px; }
  .sl .pct { width:52px; text-align:right; color:var(--muted);
             font-variant-numeric: tabular-nums; }
  .xg { font-size:1rem; margin: 4px 0 0; }
  .xg b { font-variant-numeric: tabular-nums; }
  .note { color:#fbbf24; font-size:.85rem; margin-top:10px; }
  h3 { font-size:.95rem; margin: 18px 0 6px; color:var(--muted);
       text-transform: uppercase; letter-spacing:.04em; }
  .muted { color: var(--muted); font-size:.85rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>⚽ Predictor de partidos</h1>
  <p class="sub">Dixon-Coles + Elo · Monte Carlo · datos reales 1872–2026</p>

  <div class="card">
    <form method="get" action="/">
      <div>
        <label>Equipo A</label>
        <input type="text" name="team_a" list="teams" value="{{ team_a }}"
               placeholder="Brazil" required>
      </div>
      <div>
        <label>Equipo B</label>
        <input type="text" name="team_b" list="teams" value="{{ team_b }}"
               placeholder="Argentina" required>
      </div>
      <div>
        <label>Fase</label>
        <select name="stage">
          {% for val, txt in stages %}
          <option value="{{ val }}" {{ 'selected' if val==stage else '' }}>{{ txt }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="row-full">
        <span class="chk">
          <input type="checkbox" name="home" value="1" {{ 'checked' if home else '' }}>
          Equipo A juega en casa (no neutral)
        </span>
      </div>
      <div class="row-full">
        <button type="submit">Predecir</button>
      </div>
      <datalist id="teams">
        {% for t in teams %}<option value="{{ t }}">{% endfor %}
      </datalist>
    </form>
  </div>

  {% if error %}
    <div class="card"><span class="note">{{ error }}</span></div>
  {% endif %}

  {% if pred %}
  <div class="card">
    <p class="matchup">{{ pred.team_a }} vs {{ pred.team_b }}
       <span class="muted">· {{ stage_label }}</span></p>

    {% for bar in outcome_bars %}
    <div class="bar">
      <div class="top"><span>{{ bar.label }}</span><span>{{ '%.1f'|format(bar.pct) }}%</span></div>
      <div class="track"><div class="fill"
           style="width: {{ bar.pct }}%; background: {{ bar.color }};"></div></div>
    </div>
    {% endfor %}

    <p class="xg">Goles esperados (xG): <b>{{ '%.2f'|format(pred.xg_a) }}</b>
       {{ pred.team_a }} &nbsp;·&nbsp; <b>{{ '%.2f'|format(pred.xg_b) }}</b> {{ pred.team_b }}</p>

    <h3>Marcadores más probables</h3>
    {% for s in pred.top_scorelines %}
    <div class="sl">
      <span class="score">{{ s.score }}</span>
      <div class="track"><div class="fill" style="width: {{ s.probability*100 }}%;"></div></div>
      <span class="pct">{{ '%.1f'|format(s.probability*100) }}%</span>
    </div>
    {% endfor %}

    <h3>Mercados</h3>
    <div class="grid2">
      {% for m in markets %}
      <div class="bar">
        <div class="top"><span>{{ m.label }}</span><span>{{ '%.0f'|format(m.pct) }}%</span></div>
        <div class="track"><div class="fill"
             style="width: {{ m.pct }}%; background:#0ea5e9;"></div></div>
      </div>
      {% endfor %}
    </div>

    {% if pred.confidence_score < 0.9 %}
    <p class="note">⚠ Confianza reducida: algún equipo no aparece en los datos de
       entrenamiento, se trata como "promedio".</p>
    {% endif %}
    <p class="muted">Acuerdo entre modelos: {{ '%.0f'|format(pred.model_agreement_score*100) }}%
       · confianza de datos: {{ '%.0f'|format(pred.confidence_score*100) }}%</p>
  </div>
  {% endif %}
</div>
</body>
</html>
"""


@app.route("/")
def index():
    """Render the form and, if teams were submitted, the prediction."""
    engine, teams = _get_engine()
    team_a = (request.args.get("team_a") or "").strip()
    team_b = (request.args.get("team_b") or "").strip()
    stage = request.args.get("stage", "group")
    home = request.args.get("home") == "1"

    ctx = dict(
        teams=teams, stages=_STAGES, team_a=team_a, team_b=team_b,
        stage=stage, home=home, pred=None, error=None,
        stage_label=dict(_STAGES).get(stage, stage),
        outcome_bars=[], markets=[],
    )

    if team_a and team_b:
        if team_a.lower() == team_b.lower():
            ctx["error"] = "Elige dos equipos distintos."
        else:
            pred = engine.predict(team_a, team_b, stage=stage, neutral=not home)
            ctx["pred"] = pred
            ctx["outcome_bars"] = [
                {"label": f"Gana {pred.team_a}", "pct": pred.win_probability_a * 100,
                 "color": "var(--a)"},
                {"label": "Empate", "pct": pred.draw_probability * 100,
                 "color": "var(--d)"},
                {"label": f"Gana {pred.team_b}", "pct": pred.win_probability_b * 100,
                 "color": "var(--b)"},
            ]
            ctx["markets"] = [
                {"label": "Más de 2.5 goles", "pct": pred.over_2_5_probability * 100},
                {"label": "Ambos marcan", "pct": pred.btts_probability * 100},
                {"label": f"Portería a 0 {pred.team_a}", "pct": pred.clean_sheet_a * 100},
                {"label": f"Portería a 0 {pred.team_b}", "pct": pred.clean_sheet_b * 100},
            ]

    return render_template_string(_PAGE, **ctx)


def main() -> None:
    print("Cargando motor (la primera vez fitea/cachea; espera unos segundos)…")
    _get_engine()
    print("Listo. Abre http://127.0.0.1:5000 en tu navegador.")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
