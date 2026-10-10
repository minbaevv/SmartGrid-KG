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
