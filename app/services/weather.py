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
