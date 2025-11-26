import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

from .utils import random_location
from .config import DEFAULT_NUM_STATIONS, DEFAULT_NUM_EV, DEFAULT_NUM_CHARGERS_PER_STATION

def generate_stations(num_stations: int = DEFAULT_NUM_STATIONS) -> pd.DataFrame:
    """Create a DataFrame of charging stations with random coordinates."""
    stations = []
    for i in range(num_stations):
        lat, lon = random_location()
        stations.append({
            "station_id": f"S{i}",
            "lat": lat,
            "lon": lon,
            "num_chargers": DEFAULT_NUM_CHARGERS_PER_STATION,
            "max_power_kw": 150.0   # total station power limit (example)
        })
    return pd.DataFrame(stations)

def generate_evs(num_evs: int = DEFAULT_NUM_EV,
                 start_time: datetime = datetime.now(),
                 horizon_hours: int = 1) -> pd.DataFrame:
    """Create a DataFrame of EV arrivals."""
    evs = []
    for i in range(num_evs):
        arrival_offset = random.expovariate(1/0.5)   # average every 30 min
        arr_time = start_time + timedelta(hours=min(arrival_offset, horizon_hours))
        lat, lon = random_location()
        soc = random.uniform(0.1, 0.6)               # state‑of‑charge (10‑60 %)
        needed_kwh = (1 - soc) * random.uniform(30, 80)   # how many kWh it wants
        evs.append({
            "ev_id": f"E{i}",
            "lat": lat,
            "lon": lon,
            "arrival_time": arr_time,
            "state_of_charge": soc,
            "need_kwh": needed_kwh,
            "max_charge_rate_kw": 22.0                # typical fast charger
        })
    return pd.DataFrame(evs)

def export_csv(df: pd.DataFrame, fname: str):
    df.to_csv(fname, index=False)

if __name__ == "__main__":
    stations = generate_stations()
    evs = generate_evs()
    export_csv(stations, "data/stations.csv")
    export_csv(evs, "data/evs.csv")
    print(" Generated stations.csv & evs.csv")
