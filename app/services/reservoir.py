"""Import verified, normalized CSV; no invented public data feed."""
import csv
from pathlib import Path
from app.schemas import ReservoirPoint

def parse_reservoir_csv(path: Path) -> list[ReservoirPoint]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"observed_on", "volume_km3", "source_url"}.issubset(reader.fieldnames or []):
            raise ValueError("Expected observed_on, volume_km3, source_url columns")
        rows = [ReservoirPoint.model_validate(row) for row in reader]
    if not rows:
        raise ValueError("CSV has no observations")
    if len({row.observed_on for row in rows}) != len(rows):
        raise ValueError("Duplicate dates in CSV")
    return rows
