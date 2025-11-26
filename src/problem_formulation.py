import itertools
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from .utils import haversine

def build_variable_index(evs: pd.DataFrame,
                         stations: pd.DataFrame,
                         slots: List[int]) -> Dict[Tuple[int,int,int], int]:
    """Map (ev_index, station_index, slot) → sequential QUBO variable id."""
    var_index = {}
    idx = 0
    for e in evs.index:
        for s in stations.index:
            for t in slots:
                var_index[(e, s, t)] = idx
                idx += 1
    return var_index

def formulate_qubo(evs: pd.DataFrame,
                   stations: pd.DataFrame,
                   slot_length_min: int = 15,
                   horizon_h: int = 1,
                   w_dist: float = 1.0,
                   w_wait: float = 5.0,
                   w_power: float = 2.0) -> Tuple[np.ndarray, float]:
    
    num_slots = int(horizon_h * 60 / slot_length_min)
    slots = list(range(num_slots))

    var_index = build_variable_index(evs, stations, slots)
    N = len(var_index)
    Q = np.zeros((N, N))

    ev_coords = evs[['lat', 'lon']].values
    st_coords = stations[['lat', 'lon']].values
    dist_ev_station = np.zeros((len(evs), len(stations)))
    for e_idx, (e_lat, e_lon) in enumerate(ev_coords):
        for s_idx, (s_lat, s_lon) in enumerate(st_coords):
            dist_ev_station[e_idx, s_idx] = haversine((e_lat, e_lon), (s_lat, s_lon))

    # 1) Travel distance
    for (e, s, t), i in var_index.items():
        Q[i, i] += w_dist * dist_ev_station[e, s]**2   # squared to keep QUBO

    # 2) Waiting time: if the EV arrives later than the start of the slot,
    slot_start_times = np.array([t * slot_length_min for t in slots])   # minutes since horizon start
    ev_arrivals = np.array([(evs.loc[e, "arrival_time"] - evs["arrival_time"].min()).total_seconds()/60
                            for e in evs.index])  # mins

    for (e, s, t), i in var_index.items():
        delay = max(0, slot_start_times[t] - ev_arrivals[e])   # minutes of waiting
        Q[i, i] += w_wait * delay**2

    # 3) Power‑spike penalty.
    for s in stations.index:
        max_power = stations.loc[s, "max_power_kw"]
        for t in slots:
            # collect variable ids that belong to (s,t)
            ids = [var_index[(e, s, t)] for e in evs.index]
            for i in ids:
                # linear term: -lambda * P   (encourages lower power)
                req_kw = evs.loc[evs.index[i // (len(stations)*len(slots))], "max_charge_rate_kw"]
                Q[i, i] += -w_power * req_kw * max_power
                # quadratic term: +lambda * P_i * P_j  (penalise simultaneous high draw)
                for j in ids:
                    req_kw_j = evs.loc[evs.index[j // (len(stations)*len(slots))], "max_charge_rate_kw"]
                    Q[i, j] += w_power * req_kw * req_kw_j / (max_power**2)

    # a) Each EV assigned exactly once
    A = 1000.0   # big penalty coefficient
    for e in evs.index:
        ids = [var_index[(e, s, t)] for s in stations.index for t in slots]
        for i in ids:
            Q[i, i] += -2 * A   # linear term
            for j in ids:
                Q[i, j] += 2 * A

    # b) Station charger capacity per slot
    for s in stations.index:
        cap = stations.loc[s, "num_chargers"]
        for t in slots:
            ids = [var_index[(e, s, t)] for e in evs.index]
            for i in ids:
                Q[i, i] += -2 * A
                for j in ids:
                    Q[i, j] += 2 * A

    offset = 0.0
    return Q, offset, var_index, slots
