import numpy as np
from qiskit_aer import Aer
from qiskit_algorithms import QAOA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.optimizers import COBYLA
from qiskit_aer.primitives import Sampler
import time

def qaoa_solver(Q: np.ndarray, offset: float = 0.0,
                p: int = 2, optimizer=None, shots: int = 1024):
    print(f"[QAOA] Starting with {Q.shape[0]} variables, p={p} layers, shots={shots}")
    start_time = time.time()
    
    if optimizer is None:
        optimizer = COBYLA(maxiter=250)

    n = Q.shape[0]
    
    # Warn if problem is too large
    if n > 20:
        print(f"[QAOA] WARNING: Problem has {n} variables. QAOA simulation is slow for >20 qubits.")
        print(f"[QAOA] This may take several minutes. Consider using --method dwave instead.")
    
    try:
        # Build QuadraticProgram
        qp = QuadraticProgram()
        for i in range(n):
            qp.binary_var(name=f"x{i}")

        # Objective - handle both diagonal and off-diagonal terms
        linear = {}
        quadratic = {}
        
        for i in range(n):
            if Q[i, i] != 0:
                linear[f"x{i}"] = Q[i, i]
        
        # Qiskit handles symmetric matrices internally
        for i in range(n):
            for j in range(i+1, n):
                if Q[i, j] != 0:
                    quadratic[(f"x{i}", f"x{j}")] = Q[i, j]
        
        qp.minimize(linear=linear, quadratic=quadratic, constant=offset)
        
        print(f"[QAOA] Built QuadraticProgram with {len(linear)} linear and {len(quadratic)} quadratic terms")

        # Convert to QUBO (validates the formulation)
        qpu = QuadraticProgramToQubo()
        qubo = qpu.convert(qp)

        # Use Sampler with proper initialization
        try:
            sampler = Sampler(run_options={"shots": shots})
            print(f"[QAOA] Using Sampler with run_options")
        except TypeError:
            # Fallback for older versions
            sampler = Sampler()
            print(f"[QAOA] Using Sampler (shots controlled by backend)")
        
        # Create QAOA instance with specified number of layers
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=p)
        
        print(f"[QAOA] Running optimization with {p} QAOA layers...")

        # Wrap QAOA inside MinimumEigenOptimizer
        optimizer_qaoa = MinimumEigenOptimizer(qaoa)
        result = optimizer_qaoa.solve(qubo)

        elapsed = time.time() - start_time
        print(f"[QAOA] Optimization completed in {elapsed:.2f} seconds")

        # Extract solution
        solution = {var: int(val) for var, val in result.variables_dict.items()}
        energy = result.fval
        
        print(f"[QAOA] Best energy found: {energy:.4f}")
        if n <= 10:
            print(f"[QAOA] Solution: {solution}")
        else:
            print(f"[QAOA] Solution vector: {list(solution.values())[:10]}...")
        
        return solution, energy
    
    except Exception as e:
        print(f"[QAOA] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback: return a random feasible solution
        print(f"[QAOA] Falling back to random solution")
        solution = {f"x{i}": np.random.randint(0, 2) for i in range(n)}
        
        # Calculate energy for this random solution
        x = np.array([solution[f"x{i}"] for i in range(n)])
        energy = float(x.T @ Q @ x) + offset
        
        return solution, energy


if __name__ == "__main__":
    # Simple test with a 3-variable QUBO
    print("Testing QAOA solver with small QUBO...")
    Q_test = np.array([
        [1, -1, 0],
        [-1, 1, -1],
        [0, -1, 1]
    ])
    
    sol, eng = qaoa_solver(Q_test, offset=0.0, p=2, shots=1024)
    print(f"\nTest completed!")
    print(f"Solution: {sol}")
    print(f"Energy: {eng:.4f}")
