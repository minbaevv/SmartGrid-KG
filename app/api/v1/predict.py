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
