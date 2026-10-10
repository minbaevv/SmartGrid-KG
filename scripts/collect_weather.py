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
