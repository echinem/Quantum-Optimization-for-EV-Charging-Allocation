from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from pathlib import Path

NUM_EVS = 5
NUM_STATIONS = 2  
NUM_SLOTS = 4
QAOA_LAYERS = 2

DISPLAY_QUBITS = 8  # Show 8 qubits to keep it readable

print(f"Creating QAOA circuit visualization...")
print(f"  Problem: {NUM_EVS} EVs, {NUM_STATIONS} stations, {NUM_SLOTS} slots")
print(f"  Total qubits: {NUM_EVS * NUM_STATIONS * NUM_SLOTS}")
print(f"  Displaying: {DISPLAY_QUBITS} qubits (showing pattern)")

# Create circuit
qc = QuantumCircuit(DISPLAY_QUBITS)

# Initial superposition
qc.h(range(DISPLAY_QUBITS))
qc.barrier(label='Init')

# QAOA layers
for layer in range(QAOA_LAYERS):
    # Cost layer
    qc.barrier(label=f'Cost{layer}')
    
    for i in range(DISPLAY_QUBITS):
        gamma = Parameter(f'γ{layer}_q{i}')
        qc.rz(gamma, i)
    
    # Entanglement
    for i in range(DISPLAY_QUBITS - 1):
        qc.cx(i, i + 1)
    
    # Mixer layer  
    qc.barrier(label=f'Mix{layer}')
    
    for i in range(DISPLAY_QUBITS):
        beta = Parameter(f'β{layer}_q{i}')
        qc.rx(beta, i)

# Measurement
qc.barrier(label='Meas')
qc.measure_all()

# Draw and save
print("Drawing circuit...")
fig = plt.figure(figsize=(16, 8))

try:
    qc.draw(output='mpl', style='iqp', fold=-1, ax=fig.gca())
    
    plt.suptitle(
        f'QAOA Circuit for EV Charging\n' +
        f'{NUM_EVS} EVs × {NUM_STATIONS} Stations × {NUM_SLOTS} Slots = ' +
        f'{NUM_EVS*NUM_STATIONS*NUM_SLOTS} qubits, {QAOA_LAYERS} layers\n' +
        f'(Showing {DISPLAY_QUBITS} qubits - pattern repeats)',
        fontsize=14, fontweight='bold'
    )
    
    # Save
    output_file = Path('data/qaoa_circuit.png')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n SUCCESS! Circuit saved to: {output_file}")
    print(f"   Circuit depth: {qc.depth()}")
    print(f"   Total gates: {len(qc.data)}")
    
except Exception as e:
    print(f"\n Error: {e}")
    print("Trying simpler visualization...")
    
    # Fallback to text
    plt.close(fig)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.gca()
    
    circuit_text = qc.draw(output='text', fold=-1)
    ax.text(0.1, 0.5, str(circuit_text), 
           family='monospace', fontsize=6, verticalalignment='center')
    ax.axis('off')
    
    plt.title(f'QAOA Circuit ({DISPLAY_QUBITS} qubits, {QAOA_LAYERS} layers)')
    
    output_file = Path('data/qaoa_circuit.png')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f" Fallback image saved to: {output_file}")

print("\nNext: Run 'streamlit run app/app.py' and select QAOA!")