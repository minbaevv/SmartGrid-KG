"""Train, calibrate, test on consecutive time blocks; synthetic demo only."""
import argparse
import asyncio
import json
import pickle
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sqlalchemy import select
from app.db import SessionLocal, init_db, engine
from app.models.substation import SubstationLoad, Substation
from app.ml.features import FEATURES
from app.config import get_settings

async def read_training_data() -> pd.DataFrame:
    async with SessionLocal() as session:
        rows = (await session.execute(select(
            SubstationLoad.timestamp, SubstationLoad.temperature_c,
            SubstationLoad.humidity_pct, SubstationLoad.is_overloaded,
            SubstationLoad.substation_id, SubstationLoad.weather_source, SubstationLoad.dataset_version,
        ).join(Substation).where(SubstationLoad.is_synthetic.is_(True),
                                Substation.is_synthetic.is_(True)))).mappings().all()
    return pd.DataFrame([dict(row) for row in rows])

def train(frame: pd.DataFrame, output: Path) -> dict:
    if frame.empty:
        raise ValueError("No demo data. Run scripts.generate_synthetic_data first.")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if frame.duplicated(["substation_id", "timestamp"]).any():
        raise ValueError("Duplicate station/hour rows")
    frame = frame.sort_values(["timestamp", "substation_id"])
    local = frame["timestamp"].dt.tz_convert("Asia/Bishkek")
    frame["hour"] = local.dt.hour
    frame["day_of_week"] = local.dt.dayofweek
    x = frame[FEATURES].astype(float)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Features contain non-finite values")
    times = frame["timestamp"].drop_duplicates().to_list()
    if len(times) < 240:
        raise ValueError("At least 240 distinct hours required")
    first, second = times[int(len(times)*0.6)], times[int(len(times)*0.8)]
    masks = [frame["timestamp"] < first,
             (frame["timestamp"] >= first) & (frame["timestamp"] < second),
             frame["timestamp"] >= second]
    splits = [(x.loc[mask], frame.loc[mask,"is_overloaded"].astype(int)) for mask in masks]
    if any(y.nunique() != 2 for _, y in splits):
        raise ValueError("Every time block must contain both classes; do not fabricate metrics.")
    (x_train,y_train),(x_cal,y_cal),(x_test,y_test) = splits
    forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=12,
                                    random_state=42, n_jobs=-1)
    forest.fit(x_train,y_train)
    model = CalibratedClassifierCV(FrozenEstimator(forest), method="sigmoid")
    model.fit(x_cal,y_cal)
    probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    baseline = DummyClassifier(strategy="prior").fit(x_train,y_train)
    baseline_prob = baseline.predict_proba(x_test)[:, list(baseline.classes_).index(1)]
    blocks = []
    for label,mask,(_,y) in zip(["train","calibration","test"], masks,splits,strict=True):
        subset = frame.loc[mask,"timestamp"]
        blocks.append({"block":label,"rows":len(y),"start":subset.min().isoformat(),
            "end":subset.max().isoformat(),"overload_rate":float(y.mean())})
    metadata = {"schema_version":1,"model_mode":"synthetic_demo",
        "features":FEATURES,"timezone":"Asia/Bishkek",
        "sklearn_version":sklearn.__version__,"trained_at":datetime.now(timezone.utc).isoformat(),
        "training_rows":len(frame),"time_blocks":blocks,
        "weather_sources":frame["weather_source"].value_counts().to_dict(),
        "dataset_versions":frame["dataset_version"].value_counts().to_dict(),
        "metrics": {"test_roc_auc":float(roc_auc_score(y_test,probabilities)),
            "test_brier":float(brier_score_loss(y_test,probabilities)),
            "test_log_loss":float(log_loss(y_test,probabilities)),
            "baseline_brier":float(brier_score_loss(y_test,baseline_prob))},
        "warning":"Synthetic validation does not establish real-world blackout accuracy. "
                  "Pooled model, not district-specific or probability of any station failing."}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement: do not expose a partial pickle to concurrent API requests.
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
            temp_name=tmp.name
            pickle.dump({"model":model,"metadata":metadata},tmp,protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_name,output)
    finally:
        if temp_name and Path(temp_name).exists(): Path(temp_name).unlink()
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    return metadata

async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=get_settings().model_path)
    args=parser.parse_args()
    try:
        await init_db()
        frame=await read_training_data()
        metadata=await asyncio.to_thread(train,frame,args.output)
        print(json.dumps(metadata,ensure_ascii=False))
    finally:
        await engine.dispose()

if __name__=="__main__":
    asyncio.run(main())
