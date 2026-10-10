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
