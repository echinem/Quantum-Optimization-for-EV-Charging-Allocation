import numpy as np
import pandas as pd
import dimod
from .problem_formulation import formulate_qubo
from .utils import chunked

def greedy_scheduler(evs: pd.DataFrame,
                    stations: pd.DataFrame,
                    slot_length_min: int = 15,
                    horizon_h: int = 1):
    """Very fast deterministic heuristic – returns a dict {(ev,slot,station)}."""
    # discretise time slots
    num_slots = int(horizon_h*60/slot_length_min)
    slot_start = np.arange(num_slots) * slot_length_min
    # keep track of available chargers per station/slot
    capacity = stations["num_chargers"].values[:, None].repeat(num_slots, axis=1)

    assignments = {}
    for ev_idx, ev in evs.iterrows():
        # travel distances to each station
        dists = np.array([np.linalg.norm([ev.lat - s.lat, ev.lon - s.lon])
                          for s in stations.itertuples()])
        # sort stations by distance
        order = np.argsort(dists)

        # earliest possible slot given arrival time
        arrival_min = (ev.arrival_time - evs["arrival_time"].min()).total_seconds()/60.0
        earliest_slot = int(np.ceil(arrival_min/slot_length_min))

        for s_idx in order:
            # find first slot >= earliest_slot with free charger
            for t in range(earliest_slot, num_slots):
                if capacity[s_idx, t] > 0:
                    assignments[ev_idx] = (s_idx, t)
                    capacity[s_idx, t] -= 1
                    break
            if ev_idx in assignments:
                break
        if ev_idx not in assignments:
            # fallback – put at last slot (will be penalised later)
            assignments[ev_idx] = (order[0], num_slots-1)
    return assignments


def simulated_annealing_qbsolv(Q: np.ndarray, offset: float = 0.0,
                               num_reads: int = 100):
    """Use dimod's SimulatedAnnealingSampler on the QUBO."""
    sampler = dimod.SimulatedAnnealingSampler()
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q, offset=offset)
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    best = sampleset.first.sample
    best_energy = sampleset.first.energy
    return best, best_energy
