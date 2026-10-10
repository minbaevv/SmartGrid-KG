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
