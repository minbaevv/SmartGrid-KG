# SmartGrid KG — Этап 2: полный код

Python 3.11+. Реализация с асинхронным SQLAlchemy 2.0 и RandomForest, pandas/scikit-learn. Это **demo на синтетических нагрузках**, а не проверенная модель аварий энергосети.

Для локального AI-агента: создайте/замените файлы по заголовкам путей. Все исходники уже находятся в ZIP; повторно извлекать их из этого документа не обязательно. Поддерживающие изменения обязательны: без async app/db.py, регистрации router и зависимостей четыре основные файла не работают.

## Команды выполнения
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env
python -m scripts.collect_weather
python -m scripts.generate_synthetic_data
python -m app.ml.train
python -m unittest discover -s tests -v
uvicorn app.main:app --reload
```

Если используется только этот Markdown без ZIP, установите зависимости из requirements.txt ниже. Создайте пустые `__init__.py` в app/, app/models/, app/ml/, app/api/, app/api/v1/, app/services/, scripts/. Запускайте команды из корня проекта; Windows: активация `.venv\Scripts\Activate.ps1`, копирование `Copy-Item .env.example .env`.

Модель создаётся в app/ml/model.pkl. База и модель не включены в исходный ZIP. Проверьте GET `/api/v1/predict?district=leninsky&hours_ahead=24` через `/docs`. API поддерживает 1–48 **следующих полных часов**, начиная с ближайшего будущего часа.

Погода за прошлые три месяца в текущем прогнозе отсутствует. Генератор использует доступный архив weather_forecasts без будущих выпусков; для недостающих часов создаёт явно помеченную синтетическую погоду. Все нагрузки и оборудование вымышленные. Ни `actual_load_mw`, ни `is_overloaded` не входят в признаки модели. Все станции одного часа находятся в одном временном блоке. Модель общая для районов; процент относится к синтетической подстанции/часу, не блэкауту района.

Pickle допустим только из доверенного локального обучения. Никогда не загружайте неизвестный pickle. PostgreSQL поддержан драйвером, но проверка выполнена только на SQLite. create_all не заменяет миграции Alembic.

## app/models/substation.py

```python
from datetime import datetime
from sqlalchemy import (String, Float, Boolean, DateTime, ForeignKey,
                        UniqueConstraint, CheckConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base

class Substation(Base):
    __tablename__ = "substations"
    __table_args__ = (CheckConstraint("max_capacity_mw > 0", name="ck_positive_capacity"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    district: Mapped[str] = mapped_column(String(80), index=True)
    max_capacity_mw: Mapped[float] = mapped_column(Float)
    # Demo equipment is not a register of actual substations in Bishkek.
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    loads: Mapped[list["SubstationLoad"]] = relationship(back_populates="substation")

class SubstationLoad(Base):
    __tablename__ = "substation_loads"
    __table_args__ = (
        UniqueConstraint("substation_id", "timestamp", name="uq_station_hour"),
        CheckConstraint("actual_load_mw >= 0", name="ck_nonnegative_load"),
        CheckConstraint("humidity_pct BETWEEN 0 AND 100", name="ck_humidity"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    substation_id: Mapped[int] = mapped_column(ForeignKey("substations.id"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    actual_load_mw: Mapped[float] = mapped_column(Float)
    is_overloaded: Mapped[bool] = mapped_column(Boolean)
    # Feature snapshots prevent ambiguous joins across forecast runs.
    temperature_c: Mapped[float] = mapped_column(Float)
    humidity_pct: Mapped[float] = mapped_column(Float)
    weather_source: Mapped[str] = mapped_column(String(40))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(60), default="unknown")
    substation: Mapped[Substation] = relationship(back_populates="loads")
```

## scripts/generate_synthetic_data.py

```python
"""DEMO only: synthetic equipment, loads, and fallback weather. Python 3.11+."""
import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import select
from app.db import init_db, SessionLocal, WeatherObservation, engine
from app.models.substation import Substation, SubstationLoad
from app.ml.features import DISTRICTS

def load_ratio(temperature: float, hour: int, day_of_week: int, noise: float) -> float:
    cold = max(0.0, -temperature)
    evening_peak = np.exp(-((hour - 19) / 3) ** 2)
    return max(0.05, 0.78 + 0.17 * evening_peak + 0.008 * max(0, 18-temperature)
               + 0.0018 * cold**2 - 0.015 * (day_of_week >= 5) + noise)


def build_weather(hours: pd.DatetimeIndex, archived: list[dict], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Illustrative seasonal curve, NOT measured Bishkek climatology.
    annual = 10 + 17 * np.cos(2 * np.pi * (hours.dayofyear.to_numpy() - 200) / 365.25)
    temps = annual + 4 * np.sin(2 * np.pi * (hours.hour.to_numpy() - 8) / 24)
    temps += rng.normal(0, 2, len(hours))
    frame = pd.DataFrame({"timestamp": hours.tz_convert("UTC"),
        "temperature_c": temps,
        "humidity_pct": np.clip(65 - temps * 0.8 + rng.normal(0, 8, len(hours)), 10, 100),
        "weather_source": "synthetic_weather"})
    if archived:
        history = pd.DataFrame(archived)
        history["forecast_at"] = pd.to_datetime(history["forecast_at"], utc=True)
        history["issued_at"] = pd.to_datetime(history["issued_at"], utc=True)
        # Use only forecasts available before the target hour, never a future run.
        history = history[history["issued_at"] <= history["forecast_at"]]
        history = history.sort_values("issued_at").drop_duplicates("forecast_at", keep="last")
        if not history.empty:
            history = history.set_index("forecast_at")
            mask = frame["timestamp"].isin(history.index)
            matched = history.reindex(frame.loc[mask, "timestamp"])
            frame.loc[mask, ["temperature_c", "humidity_pct"]] = matched[
                ["temperature_c", "humidity_pct"]].to_numpy()
            frame.loc[mask, "weather_source"] = "archived_forecast"
    return frame

async def generate(end: datetime | None = None, seed: int = 42) -> dict:
    await init_db()
    end_ts = pd.Timestamp(end or datetime.now(timezone.utc))
    if end_ts.tzinfo is None:
        raise ValueError("end must include timezone")
    end_ts = end_ts.tz_convert("Asia/Bishkek").floor("h")
    # Three calendar months, excluding the current/future hour.
    start_ts = end_ts - pd.DateOffset(months=3)
    hours = pd.date_range(start_ts, end_ts, freq="h", inclusive="left")
    async with SessionLocal() as session:
        archived = (await session.execute(select(
            WeatherObservation.forecast_at, WeatherObservation.issued_at,
            WeatherObservation.temperature_c, WeatherObservation.humidity_pct
        ).where(WeatherObservation.forecast_at >= start_ts.tz_convert("UTC").isoformat(),
                WeatherObservation.forecast_at < end_ts.tz_convert("UTC").isoformat()))).mappings().all()
    weather = build_weather(hours, [dict(row) for row in archived], seed)
    rng = np.random.default_rng(seed + 1)
    created = 0
    for district in DISTRICTS:
        for number, capacity in ((1, 10.0), (2, 16.0)):
            name = f"DEMO-{district}-{number}"
            async with SessionLocal.begin() as session:
                station = await session.scalar(select(Substation).where(Substation.name == name))
                if station is None:
                    station = Substation(name=name, district=district,
                        max_capacity_mw=capacity, is_synthetic=True)
                    session.add(station)
                    await session.flush()
                elif not station.is_synthetic or station.max_capacity_mw != capacity:
                    raise ValueError(f"Refusing to alter non-matching station: {name}")
                existing = set((await session.scalars(select(SubstationLoad.timestamp).where(
                    SubstationLoad.substation_id == station.id))).all())
                existing_utc = {pd.Timestamp(t).tz_localize("UTC") if pd.Timestamp(t).tzinfo is None
                                else pd.Timestamp(t).tz_convert("UTC") for t in existing}
                for row, local_time in zip(weather.to_dict("records"), hours, strict=True):
                    temp = row["temperature_c"]
                    ratio = load_ratio(temp, local_time.hour, local_time.dayofweek,
                                       rng.normal(0, 0.065))
                    load = capacity * ratio
                    timestamp = row["timestamp"]
                    if timestamp in existing_utc:
                        continue
                    session.add(SubstationLoad(substation_id=station.id,
                        timestamp=timestamp.to_pydatetime(), actual_load_mw=load,
                        is_overloaded=load > capacity, temperature_c=temp,
                        humidity_pct=row["humidity_pct"], weather_source=row["weather_source"],
                        is_synthetic=True, dataset_version=f"demo-v0.2-seed{seed}"))
                    created += 1
    return {"dataset_mode": "synthetic_demo", "seed": seed, "created_rows": created,
        "hours": len(hours), "substations": 8,
        "start_inclusive": start_ts.isoformat(), "end_exclusive": end_ts.isoformat(),
        "weather_hour_sources": weather["weather_source"].value_counts().to_dict(),
        "warning": "Equipment and all loads are synthetic, even when weather is archived."}

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", help="ISO timestamp with timezone; default current hour")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("data/generation_report.json"))
    args = parser.parse_args()
    try:
        report = await generate(datetime.fromisoformat(args.end) if args.end else None, args.seed)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
```

## app/ml/train.py

```python
"""Train, calibrate, test on consecutive time blocks; synthetic demo only."""
import argparse
import asyncio
import json
import pickle
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sqlalchemy import select
from app.db import SessionLocal, init_db, engine
from app.models.substation import SubstationLoad, Substation
from app.ml.features import FEATURES
from app.config import get_settings

async def read_training_data() -> pd.DataFrame:
    async with SessionLocal() as session:
        rows = (await session.execute(select(
            SubstationLoad.timestamp, SubstationLoad.temperature_c,
            SubstationLoad.humidity_pct, SubstationLoad.is_overloaded,
            SubstationLoad.substation_id, SubstationLoad.weather_source, SubstationLoad.dataset_version,
        ).join(Substation).where(SubstationLoad.is_synthetic.is_(True),
                                Substation.is_synthetic.is_(True)))).mappings().all()
    return pd.DataFrame([dict(row) for row in rows])

def train(frame: pd.DataFrame, output: Path) -> dict:
    if frame.empty:
        raise ValueError("No demo data. Run scripts.generate_synthetic_data first.")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if frame.duplicated(["substation_id", "timestamp"]).any():
        raise ValueError("Duplicate station/hour rows")
    frame = frame.sort_values(["timestamp", "substation_id"])
    local = frame["timestamp"].dt.tz_convert("Asia/Bishkek")
    frame["hour"] = local.dt.hour
    frame["day_of_week"] = local.dt.dayofweek
    x = frame[FEATURES].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Features contain non-finite values")
    times = frame["timestamp"].drop_duplicates().to_list()
    if len(times) < 240:
        raise ValueError("At least 240 distinct hours required")
    first, second = times[int(len(times)*0.6)], times[int(len(times)*0.8)]
    masks = [frame["timestamp"] < first,
             (frame["timestamp"] >= first) & (frame["timestamp"] < second),
             frame["timestamp"] >= second]
    splits = [(x.loc[mask], frame.loc[mask,"is_overloaded"].astype(int)) for mask in masks]
    if any(y.nunique() != 2 for _, y in splits):
        raise ValueError("Every time block must contain both classes; do not fabricate metrics.")
    (x_train,y_train),(x_cal,y_cal),(x_test,y_test) = splits
    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=12,
                                    random_state=42, n_jobs=-1)
    forest.fit(x_train,y_train)
    model = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid")
    model.fit(x_cal,y_cal)
    probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    baseline = DummyClassifier(strategy="prior").fit(x_train,y_train)
    baseline_prob = baseline.predict_proba(x_test)[:, list(baseline.classes_).index(1)]
    blocks = []
    for label,mask,(_,y) in zip(["train","calibration","test"], masks,splits,strict=True):
        subset = frame.loc[mask,"timestamp"]
        blocks.append({"block":label,"rows":len(y),"start":subset.min().isoformat(),
            "end":subset.max().isoformat(),"overload_rate":float(y.mean())})
    metadata = {"schema_version":1,"model_mode":"synthetic_demo",
        "features":FEATURES,"timezone":"Asia/Bishkek",
        "sklearn_version":sklearn.__version__,"trained_at":datetime.now(timezone.utc).isoformat(),
        "training_rows":len(frame),"time_blocks":blocks,
        "weather_sources":frame["weather_source"].value_counts().to_dict(),
        "dataset_versions":frame["dataset_version"].value_counts().to_dict(),
        "metrics": {"test_roc_auc":float(roc_auc_score(y_test,probabilities)),
            "test_brier":float(brier_score_loss(y_test,probabilities)),
            "test_log_loss":float(log_loss(y_test,probabilities)),
            "baseline_brier":float(brier_score_loss(y_test,baseline_prob))},
        "warning":"Synthetic validation does not establish real-world blackout accuracy. "
                  "Pooled model, not district-specific or probability of any station failing."}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement: do not expose a partial pickle to concurrent API requests.
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
            temp_name=tmp.name
            pickle.dump({"model":model,"metadata":metadata},tmp,protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_name,output)
    finally:
        if temp_name and Path(temp_name).exists(): Path(temp_name).unlink()
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata

async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=get_settings().model_path)
    args=parser.parse_args()
    try:
        await init_db()
        frame=await read_training_data()
        metadata=await asyncio.to_thread(train,frame,args.output)
        print(json.dumps(metadata,ensure_ascii=False))
    finally:
        await engine.dispose()

if __name__=="__main__":
    asyncio.run(main())
```

## app/api/v1/predict.py

```python
import asyncio
import pickle
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import sklearn
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from app.db import SessionLocal
from app.models.substation import Substation
from app.config import get_settings
from app.ml.features import FEATURES, DISTRICTS, normalize_district
from app.services.weather import fetch_weather, UpstreamError

router=APIRouter(prefix="/api/v1",tags=["Prediction — synthetic demo"])

class HourRisk(BaseModel):
    timestamp: datetime
    overload_risk_pct: float = Field(ge=0,le=100)

class PredictionResponse(BaseModel):
    district: str
    district_name: str
    hours_ahead: int
    generated_at: datetime
    weather_issued_at: datetime
    model_mode: str = "synthetic_demo"
    risk_scope: str = "pooled_synthetic_station_hour"
    is_demo: bool = True
    warning: str
    forecasts: list[HourRisk]

@lru_cache(maxsize=2)
def load_model(path_string: str, modified_ns: int):
    # SECURITY: only load a locally generated/trusted artifact, never an uploaded pickle.
    with Path(path_string).open("rb") as file:
        artifact=pickle.load(file)
    metadata=artifact["metadata"]
    if (metadata.get("schema_version") != 1 or metadata.get("features") != FEATURES
        or metadata.get("model_mode") != "synthetic_demo"
        or metadata.get("sklearn_version") != sklearn.__version__):
        raise ValueError("Incompatible artifact: retrain with current dependencies")
    model=artifact["model"]
    if set(model.classes_) != {0,1}:
        raise ValueError("Model must contain both classes")
    return artifact

async def artifact_for_request():
    path=get_settings().model_path.resolve()
    try:
        stat=path.stat()
        return await asyncio.to_thread(load_model,str(path),stat.st_mtime_ns)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, pickle.UnpicklingError) as exc:
        raise HTTPException(503,"Model unavailable or incompatible; run python -m app.ml.train") from exc

def predict_rows(artifact,points) -> list[HourRisk]:
    frame=pd.DataFrame([{"temperature_c":p.temperature_c,"humidity_pct":p.humidity_pct,
        "hour":p.forecast_at.astimezone(ZoneInfo("Asia/Bishkek")).hour,
        "day_of_week":p.forecast_at.astimezone(ZoneInfo("Asia/Bishkek")).weekday()}
        for p in points],columns=FEATURES)
    model=artifact["model"]
    scores=model.predict_proba(frame)[:,list(model.classes_).index(1)]
    if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError("Invalid model scores")
    return [HourRisk(timestamp=p.forecast_at,overload_risk_pct=round(float(s)*100,2))
            for p,s in zip(points,scores,strict=True)]

@router.get("/predict",response_model=PredictionResponse)
async def predict(request: Request, district: str = Query(min_length=1,max_length=80),
                  hours_ahead: int = Query(default=24,ge=1,le=48)):
    try:
        key=normalize_district(district)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    async with SessionLocal() as session:
        count=await session.scalar(select(func.count()).select_from(Substation).where(
            Substation.district==key,Substation.is_synthetic.is_(True)))
    if not count:
        raise HTTPException(404,"No demo stations in district; run synthetic generator")
    artifact=await artifact_for_request()
    try:
        # 3 calendar days are necessary for the next 48 hours near midnight.
        weather=await fetch_weather(request.app.state.http_client,get_settings(),days=3)
    except UpstreamError as exc:
        raise HTTPException(502,str(exc)) from exc
    now=datetime.now(timezone.utc)
    start=now.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
    by_time={p.forecast_at.astimezone(timezone.utc):p for p in weather.points}
    requested=[start+timedelta(hours=i) for i in range(hours_ahead)]
    if any(t not in by_time for t in requested):
        raise HTTPException(502,"Weather provider did not supply a complete future horizon")
    try:
        risks=await asyncio.to_thread(predict_rows,artifact,[by_time[t] for t in requested])
    except (ValueError,KeyError,AttributeError) as exc:
        raise HTTPException(503,"Inference failed; retrain the local model") from exc
    return PredictionResponse(district=key,district_name=DISTRICTS[key],hours_ahead=hours_ahead,
        generated_at=now,weather_issued_at=weather.issued_at,forecasts=risks,
        warning="DEMO: synthetic loads and equipment. Same pooled model for every district. "
                "Risk is not a validated blackout probability and must not control generators.")
```

## requirements.txt

```text
fastapi>=0.115,<1
uvicorn[standard]>=0.30,<1
pydantic>=2.9,<3
pydantic-settings>=2.5,<3
httpx>=0.27,<1
pandas>=2.2,<4
scikit-learn>=1.6,<2
sqlalchemy[asyncio]>=2.0,<3
aiosqlite>=0.20,<1
asyncpg>=0.29,<1
numpy>=1.26,<3
```

## .env.example

```text
APP_NAME=SmartGrid KG
DATABASE_URL=sqlite:///./data/smartgrid.db
CORS_ORIGINS=["http://localhost:3000"]
OPEN_METEO_URL=https://api.open-meteo.com/v1/forecast
HTTP_TIMEOUT_SECONDS=15
```

## app/config.py

```python
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "SmartGrid KG"
    database_url: str = "sqlite:///./data/smartgrid.db"
    cors_origins: list[str] = ["http://localhost:3000"]
    open_meteo_url: str = "https://api.open-meteo.com/v1/forecast"
    model_path: Path = Path(__file__).resolve().parent / "ml" / "model.pkl"
    http_timeout_seconds: float = Field(default=15, gt=0, le=120)

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

## app/db.py

```python
from pathlib import Path
from sqlalchemy import String, Float, UniqueConstraint, select, text, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.engine import make_url
from app.config import get_settings
from app.schemas import WeatherForecast, ReservoirPoint
from datetime import datetime, timezone

class Base(DeclarativeBase):
    pass

class WeatherObservation(Base):
    __tablename__ = "weather_forecasts"
    __table_args__ = (UniqueConstraint("issued_at", "forecast_at", name="uq_weather_run"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    issued_at: Mapped[str] = mapped_column(String(40), index=True)
    forecast_at: Mapped[str] = mapped_column(String(40), index=True)
    temperature_c: Mapped[float] = mapped_column(Float)
    humidity_pct: Mapped[float] = mapped_column(Float)
    wind_speed_ms: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(100))

class ReservoirObservation(Base):
    __tablename__ = "reservoir_observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    observed_on: Mapped[str] = mapped_column(String(10), unique=True, index=True)
    volume_km3: Mapped[float] = mapped_column(Float)
    source_url: Mapped[str] = mapped_column(String(2000))
    imported_at: Mapped[str] = mapped_column(String(40))

url = make_url(get_settings().database_url)
backend = url.get_backend_name()
if backend == "sqlite":
    url = url.set(drivername="sqlite+aiosqlite")
    if url.database and url.database != ":memory:":
        Path(url.database).parent.mkdir(parents=True, exist_ok=True)
elif backend in {"postgres", "postgresql"}:
    url = url.set(drivername="postgresql+asyncpg")
else:
    raise ValueError("Only SQLite and PostgreSQL are supported")
engine = create_async_engine(url, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

if backend == "sqlite":
    @event.listens_for(engine.sync_engine, "connect")
    def sqlite_foreign_keys(connection, _):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

async def init_db() -> None:
    # Import before create_all so ML tables are registered in the same metadata.
    from app.models import substation  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def check_db() -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))

async def save_weather(forecast: WeatherForecast) -> int:
    async with SessionLocal.begin() as session:
        for point in forecast.points:
            session.add(WeatherObservation(
                issued_at=forecast.issued_at.isoformat(),
                forecast_at=point.forecast_at.astimezone(timezone.utc).isoformat(),
                temperature_c=point.temperature_c, humidity_pct=point.humidity_pct,
                wind_speed_ms=point.wind_speed_ms, source=forecast.source))
    return len(forecast.points)

async def save_reservoir(points: list[ReservoirPoint]) -> int:
    async with SessionLocal.begin() as session:
        for point in points:
            key = point.observed_on.isoformat()
            record = await session.scalar(select(ReservoirObservation).where(
                ReservoirObservation.observed_on == key))
            if record is None:
                record = ReservoirObservation(observed_on=key)
                session.add(record)
            record.volume_km3 = point.volume_km3
            record.source_url = point.source_url
            record.imported_at = datetime.now(timezone.utc).isoformat()
    return len(points)
```

## app/ml/features.py

```python
FEATURES = ["temperature_c", "humidity_pct", "hour", "day_of_week"]
DISTRICTS = {"leninsky": "Ленинский", "oktyabrsky": "Октябрьский",
             "pervomaysky": "Первомайский", "sverdlovsky": "Свердловский"}

ALIASES = {**{k: k for k in DISTRICTS},
           **{v.lower(): k for k, v in DISTRICTS.items()}}

def normalize_district(value: str) -> str:
    key = ALIASES.get(value.strip().casefold())
    if key is None:
        raise ValueError("Supported districts: " + ", ".join(DISTRICTS))
    return key
```

## app/main.py

```python
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from app.config import get_settings
from app.db import init_db, check_db
from app.api.v1.endpoints import router
from app.api.v1.predict import router as predict_router

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        app.state.http_client = client
        try:
            yield
        finally:
            from app.db import engine
            await engine.dispose()

app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_credentials=False, allow_methods=["GET"],
                   allow_headers=["*"])
app.include_router(router)
app.include_router(predict_router)

@app.get("/healthcheck", tags=["Health"])
async def healthcheck():
    try:
        await check_db()
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    return {"status": "ok", "stage": 2, "model_artifact_present": settings.model_path.exists(), "mode": "synthetic_demo"}
```

## scripts/collect_weather.py

```python
import asyncio
import httpx
from app.config import get_settings
from app.db import init_db, save_weather, engine
from app.services.weather import fetch_weather

async def main():
    settings = get_settings()
    await init_db()
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        forecast = await fetch_weather(client, settings)
    print(f"Saved {await save_weather(forecast)} forecast points")

if __name__ == "__main__":
    async def run():
        try:
            await main()
        finally:
            await engine.dispose()
    asyncio.run(run())
```

## scripts/import_reservoir.py

```python
import argparse
import asyncio
from pathlib import Path
from app.db import init_db, save_reservoir, engine
from app.services.reservoir import parse_reservoir_csv

async def main():
    parser = argparse.ArgumentParser(description="Import verified reservoir CSV")
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    points = parse_reservoir_csv(args.csv)
    await init_db()
    print(f"Imported {await save_reservoir(points)} observations")

if __name__ == "__main__":
    async def run():
        try:
            await main()
        finally:
            await engine.dispose()
    asyncio.run(run())
```

## app/services/weather.py

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import httpx
from app.config import Settings
from app.schemas import WeatherForecast, WeatherPoint

class UpstreamError(Exception):
    pass

async def fetch_weather(client: httpx.AsyncClient, settings: Settings,
                        days: int = 2) -> WeatherForecast:
    if not 1 <= days <= 7:
        raise ValueError("days must be 1..7")
    try:
        response = await client.get(settings.open_meteo_url, params={
            "latitude": 42.8746, "longitude": 74.5698,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "forecast_days": days, "timezone": "Asia/Bishkek",
            "temperature_unit": "celsius", "wind_speed_unit": "ms",
        })
        response.raise_for_status()
        raw = response.json()
        hourly = raw["hourly"]
        keys = ("time", "temperature_2m", "relative_humidity_2m", "wind_speed_10m")
        arrays = [hourly[k] for k in keys]
        if not arrays[0] or len({len(a) for a in arrays}) != 1:
            raise ValueError("Empty or mismatched hourly arrays")
        points = [WeatherPoint(
            forecast_at=datetime.fromisoformat(t).replace(tzinfo=ZoneInfo("Asia/Bishkek")),
            temperature_c=temp, humidity_pct=humidity, wind_speed_ms=wind,
        ) for t, temp, humidity, wind in zip(*arrays, strict=True)]
        if any(a.forecast_at >= b.forecast_at for a, b in zip(points, points[1:])):
            raise ValueError("Unordered hourly timestamps")
        return WeatherForecast(issued_at=datetime.now(timezone.utc), points=points)
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        raise UpstreamError("Weather provider unavailable or invalid response") from exc
```

## app/schemas.py

```python
from datetime import datetime, date
from pydantic import BaseModel, Field

class WeatherPoint(BaseModel):
    forecast_at: datetime
    temperature_c: float
    humidity_pct: float = Field(ge=0, le=100)
    wind_speed_ms: float = Field(ge=0)

class WeatherForecast(BaseModel):
    city: str = "Bishkek"
    latitude: float = 42.8746
    longitude: float = 74.5698
    source: str = "Open-Meteo"
    timezone: str = "Asia/Bishkek"
    issued_at: datetime
    points: list[WeatherPoint]

class ReservoirPoint(BaseModel):
    observed_on: date
    volume_km3: float = Field(gt=0, le=30)
    source_url: str = Field(min_length=1)
```

## app/api/v1/endpoints.py

```python
from fastapi import APIRouter, Query, Request, HTTPException
from app.config import get_settings
from app.schemas import WeatherForecast
from app.services.weather import fetch_weather, UpstreamError

router = APIRouter(prefix="/api/v1", tags=["Weather"])

@router.get("/weather", response_model=WeatherForecast)
async def weather(request: Request, days: int = Query(default=2, ge=1, le=7)):
    try:
        return await fetch_weather(request.app.state.http_client, get_settings(), days)
    except UpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
```

## app/services/reservoir.py

```python
"""Import verified, normalized CSV; no invented public data feed."""
import csv
from pathlib import Path
from app.schemas import ReservoirPoint

def parse_reservoir_csv(path: Path) -> list[ReservoirPoint]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"observed_on", "volume_km3", "source_url"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected observed_on, volume_km3, source_url columns")
        rows = [ReservoirPoint.model_validate(row) for row in reader]
    if not rows:
        raise ValueError("CSV has no observations")
    if len({row.observed_on for row in rows}) != len(rows):
        raise ValueError("Duplicate dates in CSV")
    return rows
```

## app/services/seasonality.py

```python
"""Only validated customer-supplied factors; no fictional coefficients."""
import json
import math
from pathlib import Path

def load_seasonal_coefficients(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(i) for i in range(1, 13)}
    if set(data) != expected:
        raise ValueError("Provide exactly 12 months, keys 1..12")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 for v in data.values()):
        raise ValueError("Coefficients must be positive finite numbers")
    return {int(k): float(v) for k, v in data.items()}
```

## tests/support.py

```python
import os
import tempfile
from pathlib import Path
WORK=tempfile.TemporaryDirectory()
os.environ['DATABASE_URL']=f'sqlite+aiosqlite:///{WORK.name}/tests.db'
os.environ['MODEL_PATH']=str(Path(WORK.name)/'model.pkl')
```

## tests/test_stage1.py

```python
import os, tempfile, unittest, json
from pathlib import Path
import support
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from app.main import app
from app.config import Settings
from app.db import init_db, save_weather, save_reservoir, SessionLocal, ReservoirObservation
from app.services.weather import fetch_weather, UpstreamError
from app.services.reservoir import parse_reservoir_csv
from app.services.seasonality import load_seasonal_coefficients
PAYLOAD={'hourly':{'time':['2026-10-10T00:00','2026-10-10T01:00'], 'temperature_2m':[12,11], 'relative_humidity_2m':[45,47], 'wind_speed_10m':[2,3]}}
class WeatherTests(unittest.IsolatedAsyncioTestCase):
    async def get(self, payload=PAYLOAD, status=200):
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status,json=payload))) as c:
            return await fetch_weather(c,Settings(_env_file=None))
    async def test_parse_store(self):
        result=await self.get()
        self.assertEqual(result.points[0].forecast_at.utcoffset().total_seconds(),21600)
        await init_db();self.assertEqual(await save_weather(result),2)
    async def test_upstream_error(self):
        with self.assertRaises(UpstreamError): await self.get(status=503)
    async def test_invalid_arrays(self):
        with self.assertRaises(UpstreamError):
            await self.get({'hourly':dict(PAYLOAD['hourly'], wind_speed_10m=[1])})
    async def test_days_validation(self):
        async with httpx.AsyncClient() as c:
            with self.assertRaises(ValueError): await fetch_weather(c,Settings(_env_file=None),8)
class ApiTests(unittest.TestCase):
    def test_health_docs(self):
        with TestClient(app) as c:
            self.assertEqual(c.get('/healthcheck').status_code,200)
            self.assertEqual(c.get('/healthcheck').json()['mode'],'synthetic_demo')
            self.assertEqual(c.get('/docs').status_code,200)
            self.assertEqual(c.get('/openapi.json').status_code,200)
    def test_validation_no_predict(self):
        with TestClient(app) as c:
            self.assertEqual(c.get('/api/v1/weather?days=8').status_code,422)
            self.assertEqual(c.get('/api/v1/predict').status_code,422)
class ImportTests(unittest.TestCase):
    def csv(self, contents):
        folder=tempfile.TemporaryDirectory(); self.addCleanup(folder.cleanup)
        p=Path(folder.name)/'fixture.csv';p.write_text(contents);return p
    def test_csv_and_upsert(self):
        # Synthetic test fixture, never treated as a real reservoir observation.
        rows=parse_reservoir_csv(self.csv('observed_on,volume_km3,source_url\n2026-01-01,10,https://example.org/fixture\n'))
        import asyncio
        async def store():
            await init_db();await save_reservoir(rows);await save_reservoir(rows)
            async with SessionLocal() as s:
                self.assertEqual(await s.scalar(select(func.count()).select_from(ReservoirObservation)),1)
        asyncio.run(store())
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): parse_reservoir_csv(self.csv('observed_on,volume_km3,source_url\n'))
    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            parse_reservoir_csv(self.csv('observed_on,volume_km3,source_url\n2026-01-01,10,fixture\n2026-01-01,11,fixture\n'))
    def test_seasonality(self):
        p=self.csv(json.dumps({str(i):1.0 for i in range(1,13)}))
        self.assertEqual(len(load_seasonal_coefficients(p)),12)
        p.write_text('{"1":1}')
        with self.assertRaises(ValueError): load_seasonal_coefficients(p)
if __name__=='__main__': unittest.main()
```

## tests/test_stage2.py

```python
import support
import asyncio
import pickle
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, AsyncMock
import pandas as pd
from fastapi.testclient import TestClient
from sqlalchemy import select, func
from app.main import app
from app.config import get_settings
from app.db import SessionLocal, engine
from app.models.substation import Substation, SubstationLoad
from app.schemas import WeatherForecast, WeatherPoint
from app.services.weather import UpstreamError
from scripts.generate_synthetic_data import generate, load_ratio, build_weather
from app.ml.train import read_training_data, train
from app.api.v1.predict import load_model

class Stage2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        async def setup():
            report=await generate(datetime(2026,10,10,tzinfo=timezone.utc))
            frame=await read_training_data()
            await engine.dispose()
            return report,frame
        cls.report,cls.frame=asyncio.run(setup())
        cls.path=get_settings().model_path
        cls.metadata=train(cls.frame,cls.path)
    def forecast(self,count=72):
        first=datetime.now(timezone.utc).replace(minute=0,second=0,microsecond=0)
        return WeatherForecast(issued_at=datetime.now(timezone.utc),points=[
            WeatherPoint(forecast_at=first+timedelta(hours=i),temperature_c=5,
                         humidity_pct=65,wind_speed_ms=2) for i in range(count)])
    def test_three_calendar_months_and_eight_stations(self):
        self.assertEqual(self.report['hours'],92*24)
        self.assertEqual(self.report['created_rows'],self.report['hours']*8)
        self.assertEqual(len(self.frame),self.report['created_rows'])
        self.assertFalse(self.frame.duplicated(['substation_id','timestamp']).any())
    def test_generator_is_idempotent(self):
        async def rerun():
            result=await generate(datetime(2026,10,10,tzinfo=timezone.utc))
            await engine.dispose()
            return result
        self.assertEqual(asyncio.run(rerun())['created_rows'],0)
    def test_cold_growth_is_nonlinear(self):
        a,b,c=[load_ratio(t,12,1,0) for t in [0,-5,-10]]
        self.assertGreater(b,a);self.assertGreater(c-b,b-a)
    def test_future_archived_weather_rejected(self):
        times=pd.date_range('2026-01-01',periods=2,freq='h',tz='Asia/Bishkek')
        base=build_weather(times,[],42)
        target=times[0].tz_convert('UTC')
        history=[{'forecast_at':target.isoformat(),'issued_at':(target+timedelta(hours=1)).isoformat(),
                  'temperature_c':99,'humidity_pct':99}]
        other=build_weather(times,history,42)
        self.assertEqual(other['temperature_c'].to_list(),base['temperature_c'].to_list())
    def test_label_consistency(self):
        async def check():
            async with SessionLocal() as s:
                rows=(await s.execute(select(SubstationLoad.actual_load_mw,
                    SubstationLoad.is_overloaded,Substation.max_capacity_mw).join(Substation))).all()
            await engine.dispose()
            return all(bool(label)==(load>capacity) for load,label,capacity in rows)
        self.assertTrue(asyncio.run(check()))
    def test_temporal_partitions_and_artifact(self):
        parts=self.metadata['time_blocks']
        self.assertLess(parts[0]['end'],parts[1]['start'])
        self.assertLess(parts[1]['end'],parts[2]['start'])
        self.assertEqual(sum(p['rows'] for p in parts),len(self.frame))
        artifact=load_model(str(self.path),self.path.stat().st_mtime_ns)
        self.assertEqual(artifact['metadata']['model_mode'],'synthetic_demo')
    def test_empty_data_rejected(self):
        with self.assertRaises(ValueError): train(pd.DataFrame(),self.path.parent/'empty.pkl')
    def test_predict_48_hours(self):
        with TestClient(app) as c, patch('app.api.v1.predict.fetch_weather',new=AsyncMock(return_value=self.forecast())):
            response=c.get('/api/v1/predict?district=leninsky&hours_ahead=48')
        self.assertEqual(response.status_code,200,response.text)
        data=response.json();self.assertTrue(data['is_demo'])
        self.assertEqual(len(data['forecasts']),48)
        self.assertTrue(all(0<=p['overload_risk_pct']<=100 for p in data['forecasts']))
        self.assertGreater(datetime.fromisoformat(data['forecasts'][0]['timestamp']),datetime.now(timezone.utc))
    def test_russian_district(self):
        with TestClient(app) as c, patch('app.api.v1.predict.fetch_weather',new=AsyncMock(return_value=self.forecast())):
            self.assertEqual(c.get('/api/v1/predict',params={'district':'Ленинский','hours_ahead':1}).status_code,200)
    def test_predict_invalid_inputs(self):
        with TestClient(app) as c:
            for params in [{'district':'unknown'},{'district':'leninsky','hours_ahead':0},
                           {'district':'leninsky','hours_ahead':49}]:
                self.assertEqual(c.get('/api/v1/predict',params=params).status_code,422)
    def test_missing_model(self):
        with TestClient(app) as c, patch.object(get_settings(),'model_path',self.path.parent/'missing.pkl'):
            self.assertEqual(c.get('/api/v1/predict?district=leninsky').status_code,503)
    def test_missing_weather_horizon(self):
        with TestClient(app) as c, patch('app.api.v1.predict.fetch_weather',new=AsyncMock(return_value=self.forecast(2))):
            self.assertEqual(c.get('/api/v1/predict?district=leninsky&hours_ahead=48').status_code,502)
    def test_weather_provider_failure(self):
        with TestClient(app) as c, patch('app.api.v1.predict.fetch_weather',new=AsyncMock(side_effect=UpstreamError('unavailable'))):
            self.assertEqual(c.get('/api/v1/predict?district=leninsky').status_code,502)
    def test_incompatible_artifact(self):
        p=self.path.parent/'bad.pkl'
        p.write_bytes(pickle.dumps({'metadata':{'schema_version':99}}))
        with self.assertRaises(ValueError): load_model(str(p),p.stat().st_mtime_ns)

if __name__=='__main__': unittest.main()
```
