import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# ------------------------  D‑Wave credentials  ------------------------
DWAVE_API_TOKEN = os.getenv("DWAVE_API_TOKEN", "")
DWAVE_ENDPOINT = os.getenv("DWAVE_ENDPOINT", "https://cloud.dwavesys.com/sapi")

# ------------------------  IBM Q credentials  ------------------------
IBMQ_TOKEN = os.getenv("IBMQ_TOKEN", "")

# geographic bounding box (lat, lon) – a small toy city grid
CITY_LAT_MIN, CITY_LAT_MAX = 37.75, 37.78   # e.g. San Francisco downtown
CITY_LON_MIN, CITY_LON_MAX = -122.45, -122.40

# Number of stations / chargers (adjust with command‑line args)
DEFAULT_NUM_STATIONS = 4
DEFAULT_NUM_EV = 30
DEFAULT_NUM_CHARGERS_PER_STATION = 3

# Where simulation results are saved
DATA_DIR = BASE_DIR / "data"
RESULTS_DB = BASE_DIR / "ev_charging.db"
