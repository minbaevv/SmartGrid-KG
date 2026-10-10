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
