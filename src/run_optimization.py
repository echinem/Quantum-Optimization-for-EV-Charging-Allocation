import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys
import os
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.simulate_ev_data import generate_evs, generate_stations, export_csv
from src.problem_formulation import formulate_qubo
from src.classical_optimizer import greedy_scheduler, simulated_annealing_qbsolv
from src.dwave_solver import dwave_qaoa_solver
from src.qaoa_solver import qaoa_solver
from src.db import log_run, init_db

DATA_DIR = project_root / "data"
EV_FILE = DATA_DIR / "evs.csv"
STATION_FILE = DATA_DIR / "stations.csv"

def load_data():
    evs = pd.read_csv(EV_FILE, parse_dates=["arrival_time"])
    stations = pd.read_csv(STATION_FILE)
    return evs, stations


def parse_args():
    parser = argparse.ArgumentParser(description="EV-charging optimisation demo")
    parser.add_argument("--generate-data", action="store_true",
                        help="Create synthetic EV & station CSVs")
    parser.add_argument("--method", choices=["greedy","anneal","dwave","qaoa"],
                        default="dwave",
                        help="Which solver to run (default dwave)")
    parser.add_argument("--slot-length", type=int, default=15,
                        help="Slot length in minutes")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Hours to look ahead")
    parser.add_argument("--qaoa-layers", type=int, default=2,
                        help="Number of QAOA layers (p parameter)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generate_data:
        DATA_DIR.mkdir(exist_ok=True)
        stations = generate_stations()
        evs = generate_evs()
        export_csv(stations, STATION_FILE)
        export_csv(evs, EV_FILE)
        print(" Synthetic data generated")
        return

    # Load data & build QUBO
    print(f"\n{'='*60}")
    print(f"Running {args.method.upper()} optimization")
    print(f"{'='*60}\n")
    
    evs, stations = load_data()
    print(f"Loaded {len(evs)} EVs and {len(stations)} stations")
    
    Q, offset, var_index, slots = formulate_qubo(
        evs, stations,
        slot_length_min=args.slot_length,
        horizon_h=args.horizon
    )
    if args.debug:
        print(f"QUBO size: {Q.shape[0]} variables, {len(slots)} time slots")

    # Run the chosen solver
    assignment = {}
    energy = None
    
    if args.method == "greedy":
        print("\nRunning GREEDY scheduler...")
        assignment_tuples = greedy_scheduler(
            evs, stations,
            slot_length_min=args.slot_length,
            horizon_h=args.horizon
        )
        # Convert to the format expected by the UI
        assignment = {str((k, v[0], v[1])): 1 for k, v in assignment_tuples.items()}
        energy = None
        
    elif args.method == "anneal":
        print("\nRunning SIMULATED ANNEALING...")
        sample, energy = simulated_annealing_qbsolv(Q, offset)
        # Convert binary solution to assignment
        assignment = {}
        for (e, s, t), idx in var_index.items():
            if idx in sample and sample[idx] == 1:
                assignment[str((e, s, t))] = 1
                
    elif args.method == "dwave":
        print("\nRunning D-WAVE solver...")
        sample, energy = dwave_qaoa_solver(Q, offset)
        # Convert binary solution to assignment
        assignment = {}
        for (e, s, t), idx in var_index.items():
            if idx in sample and sample[idx] == 1:
                assignment[str((e, s, t))] = 1
                
    elif args.method == "qaoa":
        print("\nRunning QAOA solver...")
        # QAOA returns a dictionary with variable names
        sol, energy = qaoa_solver(Q, offset, p=args.qaoa_layers, shots=1024)
        
        # Convert the solution to assignment format
        assignment = {}
        for (e, s, t), idx in var_index.items():
            var_name = f"x{idx}"
            if var_name in sol and sol[var_name] == 1:
                assignment[str((e, s, t))] = 1
        
    else:
        raise ValueError("Unknown method")

    # Persist run information (optional)
    try:
        init_db()
        log_run(
            method=args.method, 
            energy=energy if energy is not None else -1,
            notes=f"{args.method} run on {datetime.now().isoformat()}"
        )
    except Exception as db_error:
        print(f"Warning: Could not log to database: {db_error}")

    # Store results for the dashboard
    DATA_DIR.mkdir(exist_ok=True)
    results_path = DATA_DIR / f"result_{args.method}.json"
    
    result_data = {
        "assignment": assignment,
        "energy": energy if energy is not None else "N/A",
        "method": args.method,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "slot_length_min": args.slot_length,
            "horizon_hours": args.horizon,
            "qaoa_layers": args.qaoa_layers if args.method == "qaoa" else None
        }
    }
    
    with open(results_path, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f" {args.method.upper()} optimization completed!")
    print(f"{'='*60}")
    print(f"Energy: {energy if energy is not None else 'N/A'}")
    print(f"Assignments: {len(assignment)}")
    print(f"Results saved to: {results_path}")
    
    print(f"\nRun the dashboard to visualize results:")
    print(f"  streamlit run app/app.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()