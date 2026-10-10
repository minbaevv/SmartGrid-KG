FEATURES = ["temperature_c", "humidity_pct", "hour", "day_of_week"]
DISTRICTS = {"leninsky": "Ленинский", "oktyabrsky": "Октябрьский",
             "pervomaysky": "Первомайский", "sverdlovsky": "Свердловский"}

ALIASES = {**{k: k for k in DISTRICTS},
           **{v.lower(): k for k, v in DISTRICTS.items()}}

def normalize_district(value: str) -> str:
    key = ALIASES.get(value.strip().casefold())
    if key is None:
        raise ValueError("Supported districts: " + ", ".join(DISTRICTS))
    return key
