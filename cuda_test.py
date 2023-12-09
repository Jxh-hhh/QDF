from qiskit import transpile, execute, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
import time


qc = QFT(40)
qc = qc.compose(QFT(30, inverse=True), list(range(30)))
sim = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
tqc = transpile(qc, sim)
tqc.measure_all()
t = time.time()
res = sim.run(tqc, shots=30000)
print(res.result())
print(time.time() - t)


