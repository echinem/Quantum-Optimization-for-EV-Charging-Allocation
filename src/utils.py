import random
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from typing import Tuple, List, Dict

def random_location() -> Tuple[float, float]:
    """Return a random (lat, lon) inside the bounding box."""
    from .config import CITY_LAT_MIN, CITY_LAT_MAX, CITY_LON_MIN, CITY_LON_MAX
    lat = random.uniform(CITY_LAT_MIN, CITY_LAT_MAX)
    lon = random.uniform(CITY_LON_MIN, CITY_LON_MAX)
    return lat, lon

def haversine(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    """Straight‑line distance in km between two (lat,lon)."""
    return geodesic(a, b).km

def distance_matrix(locations: List[Tuple[float,float]]) -> np.ndarray:
    """NxN matrix of pairwise haversine distances."""
    n = len(locations)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = haversine(locations[i], locations[j])
            dm[i, j] = dm[j, i] = d
    return dm

def chunked(iterable, size):
    """Yield successive chunks of length `size`."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]
