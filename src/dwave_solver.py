import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from .problem_formulation import formulate_qubo
import time
import numpy as np

def dwave_qaoa_solver(Q, offset=0.0, num_reads=10, chain_strength=5.0, dev_mode=True):
    print("Q shape:", Q.shape)
    print("Number of nonzero elements:", np.count_nonzero(Q))
    if dev_mode and num_reads > 200:
        print(f"[INFO] Capping num_reads from {num_reads} → 200 (dev mode)")
        num_reads = 200

    # Build BQM
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)
    bqm.offset += offset
    bqm = bqm.change_vartype(dimod.BINARY)

    sampler = EmbeddingComposite(DWaveSampler()) 
    # Run solver with timing
    #sampler = dimod.SimulatedAnnealingSampler()
    start = time.time()
    response = sampler.sample(bqm, num_reads=num_reads)
    elapsed = time.time() - start
    print(f"[INFO] Annealing finished in {elapsed:.2f} seconds (num_reads={num_reads})")

    # Extract best solution
    best_sample = response.first.sample
    best_energy = response.first.energy
    return best_sample, best_energy
