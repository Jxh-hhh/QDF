from qvm import Qvm
from numpy import pi
from qiskit import QuantumCircuit


qc = QuantumCircuit(20)
qvm = Qvm('cpuOnly')
cut_config = {}
qcs = Qvm.cut(qc, cut_config)
res = Qvm.run(qc)

print(res)
