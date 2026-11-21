# Mundial Handicap ⚽

Motor de predicción probabilística para partidos de fútbol de selecciones
(enfocado en la Copa del Mundo). A partir de resultados históricos reales,
ajusta dos modelos complementarios, simula cada partido por Monte Carlo y
entrega probabilidades calibradas de victoria / empate / derrota, marcadores
más probables y mercados secundarios (más/menos goles, ambos marcan, etc.).

```
datos reales → Dixon-Coles + Elo → Monte Carlo (100k sims) → mezcla convexa → MatchPrediction
                                         ↑
                    Backtester (validación temporal sin fuga de datos)
```

## Características

- **Datos reales** de resultados internacionales (dataset `martj42/international_results`,
  licencia CC0) descargados automáticamente la primera vez.
- **Modelo Dixon-Coles** (Poisson bivariado) con decaimiento temporal,
  regularización y *partial pooling* por confederación.
- **Elo dinámico** con factor K según la importancia del partido.
- **Simulación Monte Carlo** vectorizada: resultados, marcadores, mercados
  secundarios y, en eliminatorias, prórroga y penales.
- **Mezcla de modelos** con un solo parámetro ajustado sobre un conjunto de
  validación sin fuga de información.
- **Back-test** sobre Mundiales reales (2010–2022) con log loss, Brier, RPS e
  intervalos de confianza por bootstrap.
- **Interfaz web** (Flask) con gráficos de barras y banderas de los países.
- **CLI** para predicciones instantáneas (el modelo ajustado se guarda en caché).
- Utilidades para quitar el margen de las casas de apuestas (*de-vig*) y
  combinar probabilidades de mercado.

## Requisitos

- Python 3.10 o superior
- Conexión a internet solo en la primera ejecución (para descargar el dataset)

## Instalación

```bash
git clone https://github.com/Hellum-d/Mundial-Handicap.git
cd Mundial-Handicap
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r football_predictor/requirements.txt
```

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio.

**Interfaz web** — abre http://127.0.0.1:5000

```bash
python -m football_predictor.webapp
```

**Predicción rápida por consola**

```bash
python -m football_predictor.predict "Brazil" "Argentina" --stage final
python -m football_predictor.predict "Colombia" "Spain" --home   # Colombia como local
```

Los nombres de los equipos van en inglés, tal como aparecen en el dataset.

**Back-test completo + ejemplo de predicción**

```bash
python -m football_predictor
```

**(Re)descargar el dataset**

```bash
python -m football_predictor.data.real_data
```

**Tests** (funcionan sin conexión, usan datos sintéticos)

```bash
python -m pytest football_predictor/tests -q
```

La primera predicción ajusta el modelo y lo guarda en
`football_predictor/data/engine.joblib`; las siguientes son instantáneas. Los
archivos de caché (dataset y modelo) no se suben al repositorio y se regeneran
solos.

## Estructura del proyecto

```
football_predictor/
├── config.py              # Hiperparámetros y rutas (todo lo ajustable está aquí)
├── __main__.py            # Demo: back-test + predicción de ejemplo
├── predict.py             # CLI de predicción
├── webapp.py              # Interfaz web Flask
├── market.py              # De-vig y mezcla con probabilidades de mercado
├── data/                  # Carga del dataset real, datos sintéticos, banderas, confederaciones
├── models/                # Dixon-Coles y Elo
├── simulation/            # Motor Monte Carlo
├── output/                # Motor de predicción y esquemas Pydantic
├── evaluation/            # Métricas y back-tester
└── tests/                 # Suite de tests (pytest)
```

## Configuración

Los parámetros principales están en [`football_predictor/config.py`](football_predictor/config.py):

| Parámetro | Valor | Descripción |
|---|---|---|
| `TRAIN_WINDOW_YEARS` | 2 | Años de historia usados para entrenar (prioriza la forma reciente) |
| `DIXON_COLES_XI` | 0.0019 | Decaimiento temporal (~1 año de vida media) |
| `ELO_HOME_ADVANTAGE` | 65 | Puntos Elo extra para el local |

Si cambias la configuración, borra `football_predictor/data/engine.joblib`
para forzar que el modelo se vuelva a entrenar.

## Resultados

En los Mundiales 2010–2022 usados como prueba, la mezcla de modelos obtiene un
log loss medio de **~1.00**, frente a **~1.02** del Elo solo y **1.10** de la
predicción uniforme (1/3, 1/3, 1/3). La mejora frente al uniforme es
estadísticamente significativa; la mejora frente al Elo solo es real como
estimación puntual pero no significativa con 256 partidos.

El objetivo del proyecto es producir probabilidades **bien calibradas**, no
"ganarle" a las casas de apuestas. Más detalles técnicos, decisiones de diseño
y hoja de ruta en [`football_predictor/README.md`](football_predictor/README.md).

## Datos

Dataset [`martj42/international_results`](https://github.com/martj42/international_results)
(CC0-1.0, dominio público). Las banderas se cargan desde flagcdn.com.

## Aviso

Proyecto con fines educativos y de análisis. Las predicciones son
probabilidades, no garantías: no lo uses como consejo de apuestas.
