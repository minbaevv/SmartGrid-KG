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
