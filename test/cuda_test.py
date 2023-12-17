from qiskit import transpile, execute, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import time


n_qubits = 35
qc = QFT(n_qubits)
qc = qc.compose(QFT(n_qubits, inverse=True), list(range(n_qubits)))
sim = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
# sim = AerSimulator()
tqc = transpile(qc, sim)
tqc.measure_all()
t = time.time()
res = sim.run(tqc, shots=30000)
print(res.result())
print(time.time() - t)
