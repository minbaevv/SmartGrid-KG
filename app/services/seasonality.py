"""Only validated customer-supplied factors; no fictional coefficients."""
import json
import math
from pathlib import Path

def load_seasonal_coefficients(path: Path) -> dict[int, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {str(i) for i in range(1, 13)}
    if set(data) != expected:
        raise ValueError("Provide exactly 12 months, keys 1..12")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 for v in data.values()):
        raise ValueError("Coefficients must be positive finite numbers")
    return {int(k): float(v) for k, v in data.items()}
