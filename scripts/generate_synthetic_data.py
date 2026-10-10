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
