<div align="center">

# SmartGrid KG

**Weather-driven ML API that estimates local grid overload risk in Bishkek, Kyrgyzstan — up to 48 hours ahead.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.143-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ML: scikit-learn](https://img.shields.io/badge/ML-RandomForest-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![SQLAlchemy 2.0](https://img.shields.io/badge/SQLAlchemy-2.0%20async-D71F00)](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
[![Tests](https://img.shields.io/badge/tests-24%20passing-brightgreen)](#testing)
[![CI](https://github.com/minbaevv/SmartGrid-KG/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/minbaevv/SmartGrid-KG/actions/workflows/ci.yml)
[![Status: synthetic demo](https://img.shields.io/badge/status-synthetic%20demo-orange)](#limitations)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

> [!WARNING]
> **Demo status.** The overload model in this repository is trained on **synthetic** loads and fictional
> substations, and the weather rows come from a free public forecast API. It does **not** predict real
> blackouts, does **not** know the real Bishkek grid, and must **not** be used to control generators or
> make operational decisions. See [Limitations](#limitations) for the full list.

## Problem Statement

Bishkek and the wider Kyrgyz energy system face a recurring pattern: cold-season demand spikes
(electric heating during winter peaks), a grid inherited from the Soviet era with limited spare
capacity, and fast urban growth that keeps adding load to specific districts. When a substation is
pushed past its rating for hours, the result is equipment stress, local outages and expensive
emergency repairs.

Operators usually learn about an overload when it happens. What is missing is an early, cheap signal:
*given tomorrow's weather, which districts are most likely to see overloaded stations, and when?*
That question is answerable with machine learning **if** reliable load and outage data exists — which
is exactly the hard part in this region (see [Limitations](#limitations)).

## Solution Overview

SmartGrid KG is a local-first FastAPI service that:

1. Pulls hourly weather forecasts (temperature, humidity, wind) for Bishkek from **Open-Meteo**
   (free, no API key) and stores them in SQLite/PostgreSQL.
2. Builds a feature set from weather + local calendar time (`hour`, `day_of_week`, Asia/Bishkek)
   and trains a **sigmoid-calibrated RandomForest** classifier to estimate overload risk per station-hour.
3. Serves hourly risk percentages for the **next 1–48 full hours** per district via
   `GET /api/v1/predict`, together with explicit demo labels and scope warnings.

Every prediction carries machine-readable honesty flags: `is_demo: true`, `model_mode: "synthetic_demo"`,
`risk_scope: "pooled_synthetic_station_hour"` and a human-readable warning. This is deliberate:
the pipeline is real, the data is not.

## Architecture & Tech Stack

| Layer | Technology | Status |
|---|---|---|
| API framework | FastAPI + Pydantic v2, `/docs` (Swagger UI) | ✅ implemented |
| Database | SQLAlchemy 2.0 **async**, SQLite via `aiosqlite` (default), PostgreSQL via `asyncpg` (driver supported, only SQLite verified) | ✅ implemented |
| Weather data | Open-Meteo forecast API (temperature `2m`, relative humidity, wind `10m`) | ✅ implemented |
| ML | scikit-learn `RandomForestClassifier` + sigmoid calibration (`CalibratedClassifierCV` over `FrozenEstimator`), `DummyClassifier` baseline | ✅ implemented (synthetic demo data) |
| Data generation | Idempotent 3-month synthetic load generator for 8 fictional substations | ✅ implemented (demo only) |
| Tests | `unittest` — 24 tests (weather client, API, storage, generator, training, prediction) | ✅ passing |
| Docker | `python:3.11-slim` image + `docker-compose.yml` (bind-mounted DB and model) | 🟡 added (Stage 3); image build not yet run locally |
| CI | GitHub Actions: `unittest` suite on Python 3.11 and 3.13 for every push / PR to `main` | ✅ implemented |
| Auth, rate limiting, migrations (Alembic), deployment | — | ⛔ planned (Stage 3) |
| Gradient boosting (XGBoost / LightGBM) | — | ⛔ candidate after real labelled data |

### Data flow

```text
Open-Meteo ──► app/services/weather.py ──► weather_forecasts (SQLite/PostgreSQL)
                       │
Demo: scripts/generate_synthetic_data.py ──► substation_loads (synthetic, labelled)
                       │
        app/ml/train.py (temporal 60/20/20 split) ──► app/ml/model.pkl (+ metadata.json)
                       │
GET /api/v1/predict ──► live weather + model ──► hourly overload risk, 1–48 h ahead
```

### Repository layout

```text
SmartGrid-KG/
├── app/
│   ├── api/v1/        # /weather and /predict routers
│   ├── ml/            # FEATURES, training entry point; model.pkl is generated, git-ignored
│   ├── models/        # Substation, SubstationLoad (async ORM)
│   ├── services/      # Open-Meteo client, reservoir CSV import, seasonality validator
│   ├── config.py      # pydantic-settings, .env support
│   ├── db.py          # async engine, session, weather/reservoir persistence
│   └── main.py        # FastAPI app, lifespan, /healthcheck
├── scripts/
│   ├── collect_weather.py          # real Open-Meteo forecast -> DB
│   ├── generate_synthetic_data.py  # DEMO loads for 8 fictional substations
│   └── import_reservoir.py         # verified reservoir CSV import (manual data only)
├── tests/             # 24 unit/integration tests
├── data/              # reservoir template; local SQLite DB is git-ignored
├── .github/workflows/ci.yml # CI: 24 unittest cases on Python 3.11 + 3.13
├── Dockerfile         # python:3.11-slim image, uvicorn on ${PORT:-8000}
├── docker-compose.yml # api service: :8000 + volumes for data/ and model.pkl
├── .dockerignore      # venv/, .git/, *.db, .env stay out of the image
├── requirements.txt   # direct dependencies
├── requirements.lock.txt  # pinned, reproducible install (Windows + Linux)
├── plan.md            # staged roadmap (Russian; docs are partly in Russian)
└── CHANGELOG.md
```

## Quick Start

**Prerequisites:** Python 3.11+ (developed and tested on 3.13) and internet access for the weather API.
No API keys are required.

```bash
git clone https://github.com/minbaevv/SmartGrid-KG.git
cd SmartGrid-KG

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.lock.txt

cp .env.example .env             # Windows: copy .env.example .env
```

Run the demo pipeline (all commands from the repository root):

```bash
# 1) Real Open-Meteo forecast -> local SQLite (needs internet).
python -m scripts.collect_weather

# 2) DEMO DATA: 3 months of synthetic loads for 8 fictional substations.
python -m scripts.generate_synthetic_data

# 3) Train + calibrate the demo model -> app/ml/model.pkl
python -m app.ml.train

# 4) Start the API
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/docs> for interactive Swagger UI.

### Docker (optional)

```bash
# 1) Train the model on the host first: app/ml/model.pkl is mounted into the image.
python -m app.ml.train

# 2) Build and start
docker compose up --build

# or without Compose
docker build -t smartgrid-kg .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" -v "$PWD/app/ml/model.pkl:/app/app/ml/model.pkl:ro" smartgrid-kg
```

The image is built from `python:3.11-slim` with `requirements.txt`. All runtime dependencies
(including `numpy`, `pandas`, `scikit-learn`, `asyncpg`) resolve to manylinux wheels for CPython 3.11,
so no compiler toolchain is installed. The container listens on `${PORT:-8000}` (works with PaaS that
inject `PORT`), exposes `/healthcheck` as a Docker healthcheck, and gets two bind mounts from
`docker-compose.yml`: `./data` (SQLite database survives restarts) and `./app/ml/model.pkl` (trained
model, read-only).

> [!IMPORTANT]
> The API compares the artifact's recorded scikit-learn version with the installed one and answers
> `503 Model unavailable or incompatible` on a mismatch. `requirements.txt` resolves to scikit-learn
> **1.9.1** for CPython 3.11 on Linux (checked by wheel resolution against CPython 3.11 platform tags,
> not by a container build) — the same version the local `model.pkl` was trained with — so the mounted
> artifact passes the version check. The one step that cannot be verified without Docker is unpickling
> a Python 3.13 artifact under Python 3.11; if the model is still rejected, train inside the container
> after the first start:
>
> ```bash
> docker compose exec api python -m scripts.generate_synthetic_data
> docker compose exec api python -m app.ml.train
> ```

### Testing

```bash
python -m unittest discover -s tests -v
# Ran 24 tests ... OK
```

### Continuous Integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the suite on every push and pull request
to `main`:

- matrix over Python **3.11** (the interpreter the Docker image uses) and **3.13** (the local dev
  interpreter) on `ubuntu-latest`, with the pip cache keyed on `requirements.txt`;
- `python -m pip install -r requirements.txt`, then `python -m unittest discover -s tests -v`.

The `-s tests` flag is required: the test modules import the shared helper `tests/support.py`, so
discovery has to start **inside** `tests/`. A bare `python -m unittest discover` from the repository
root finds 0 tests and still exits 0, so it must not be used in CI — it would report success without
running anything.

### Example training output (synthetic test block)

```text
test_roc_auc: 0.908 | test_brier: 0.039 | baseline_brier: 0.042
training_rows: 17664 | model_mode: synthetic_demo
```

These numbers were produced on **synthetic** data and say nothing about real-world accuracy. They are
printed so the pipeline is inspectable, not as a performance claim.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthcheck` | Liveness + DB connectivity + model artifact presence |
| `GET` | `/api/v1/weather?days=1..7` | Live Open-Meteo forecast for Bishkek (read-only; no DB writes) |
| `GET` | `/api/v1/predict?district=…&hours_ahead=1..48` | Hourly overload-risk estimate for the next full hours |

Districts: `leninsky`, `oktyabrsky`, `pervomaysky`, `sverdlovsky` (Russian names such as `Ленинский`
are accepted too). Error codes: `422` invalid district or parameters, `404` no demo stations in the
district, `503` model missing/incompatible or inference failed, `502` weather provider unavailable or
incomplete future horizon.

### `GET /healthcheck`

```json
{
  "status": "ok",
  "stage": 2,
  "model_artifact_present": true,
  "mode": "synthetic_demo"
}
```

### `GET /api/v1/weather?days=1`

Real response, truncated (24 points returned for `days=1`, up to 7 days):

```json
{
  "city": "Bishkek",
  "latitude": 42.8746,
  "longitude": 74.5698,
  "source": "Open-Meteo",
  "timezone": "Asia/Bishkek",
  "issued_at": "2026-10-10T17:31:11.580428Z",
  "points": [
    {"forecast_at": "2026-10-10T00:00:00+06:00", "temperature_c": 14.0, "humidity_pct": 51.0, "wind_speed_ms": 0.66},
    {"forecast_at": "2026-10-10T01:00:00+06:00", "temperature_c": 13.0, "humidity_pct": 55.0, "wind_speed_ms": 0.07},
    "…24 hourly points total"
  ]
}
```

### `GET /api/v1/predict?district=leninsky&hours_ahead=24`

Real response, truncated (24 forecast items returned):

```json
{
  "district": "leninsky",
  "district_name": "Ленинский",
  "hours_ahead": 24,
  "generated_at": "2026-10-10T17:31:11.989162Z",
  "weather_issued_at": "2026-10-10T17:31:11.989124Z",
  "model_mode": "synthetic_demo",
  "risk_scope": "pooled_synthetic_station_hour",
  "is_demo": true,
  "warning": "DEMO: synthetic loads and equipment. Same pooled model for every district. Risk is not a validated blackout probability and must not control generators.",
  "forecasts": [
    {"timestamp": "2026-10-11T00:00:00+06:00", "overload_risk_pct": 0.45},
    {"timestamp": "2026-10-11T01:00:00+06:00", "overload_risk_pct": 0.45},
    "…21 more hours"
  ]
}
```

`overload_risk_pct` is the calibrated probability that a **synthetic** substation-hour is overloaded.
It is pooled across districts (one shared model), not a district-specific or blackout probability.

## Demo Dataset & Model

**Synthetic loads.** `scripts/generate_synthetic_data.py` creates 8 fictional substations
(2 per district, 10/16 MW demo ratings) and 3 calendar months of hourly loads from a nonlinear
temperature model (evening peak + accelerated growth below 0 °C), using archived forecasts when
available and clearly labelled synthetic weather otherwise. The generator is idempotent and refuses
to modify stations it did not create.

**Model.** Features are `temperature_c`, `humidity_pct`, `hour`, `day_of_week` (Asia/Bishkek).
`actual_load_mw` and `is_overloaded` are never used as inputs (no target leakage). Training uses
temporal blocks (60% train / 20% calibration / 20% test); calibration is sigmoid on a frozen forest;
a `DummyClassifier(prior)` serves as a baseline. The artifact is written atomically, and the API
validates its schema, feature list, model mode and scikit-learn version before serving.

## Limitations

- **Synthetic data.** All loads and substations are invented. No real outage history exists in this
  repository, so no real-world blackout metric can be claimed. Synthetic validation does not
  establish operational accuracy.
- **Not a blackout predictor.** Risk refers to a synthetic station-hour, pooled across districts;
  it is not a probability that a district blacks out and must not drive automatic actions.
- **Reservoir (Toktogul) data.** No stable public machine-readable source is confirmed; the importer
  accepts only manually prepared, verified CSV (`observed_on, volume_km3, source_url`). The template
  is intentionally empty — no fabricated measurements or URLs.
- **PostgreSQL.** Supported by the async driver code path but only SQLite was tested; `create_all`
  is not a substitute for Alembic migrations in production.
- **No security hardening.** No authentication, rate limiting, or deployment configuration yet.
- Weather and reservoir volume alone are insufficient for real outage forecasting; real load and
  outage data with usage rights are the blocking requirement (see `plan.md`, Stage 2).

## Roadmap

Per [`plan.md`](plan.md) (Russian): **Stage 3** — Dockerfile + Compose ✅ (containerization done;
image build still to be verified on a machine with Docker), GitHub Actions CI ✅, authentication,
rate limiting, Alembic
migrations, scheduled collection, deployment (Render/Railway), client-facing docs. **Stage 4** — first
pilot with a B2B partner, quality metrics and acceptance criteria. Also planned: obtaining real,
licensed load/outage data, followed by district-specific models and gradient boosting (e.g. XGBoost)
experiments once labelled real data exists.

## Data Sources, Attribution & License

- Weather data: [Open-Meteo](https://open-meteo.com/) — free for non-commercial use; check
  [terms](https://open-meteo.com/en/terms) and [pricing](https://open-meteo.com/en/pricing) before any
  commercial use. Free access does not imply unlimited commercial rights.
- Reservoir volume publication limits in Kyrgyzstan are discussed publicly
  [here](https://24.kg/obschestvo/330907_dannyie_obnbspobyeme_vodyi_vnbsptoktogulskom_vodohranilische_teper_zasekrechenyi/)
  (news article, not a data source).
- Code license: [MIT](LICENSE). This is a local prototype, not a production system or a power-grid
  control tool.
