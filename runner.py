from utils import *




def run_on_cpu(qc: QuantumCircuit, shots=1024):
    """
    开一个进程在CPU跑电路，开进程开销也比较大，谨慎；等能跑通后，看线程咋样了
    :return:
    """
    pass

def run_on_gpu(qc: QuantumCircuit, shots=1024):
    """
    开一个进程在GPU跑电路，开进程开销也比较大，谨慎；等能跑通后，看线程咋样了
    :return:
    """
    pass

def run_on_qpu(qc: QuantumCircuit, shots=1024):
    """
    开一个进程在QPU跑电路，开进程开销也比较大，谨慎；等能跑通后，看线程咋样了
    :return:
    """
    pass

@ray.remote
def run_circuit(qc: QuantumCircuit, type="QPU", shots=1024):
    if type == "QPU":
        return run_on_qpu(qc, shots)
    elif type == "CPU":
        return run_on_cpu(qc, shots)
    elif type == "GPU":
        return run_on_gpu(qc, shots)
    else:
        raise Exception("没有这种运行方式")
