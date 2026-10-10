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
