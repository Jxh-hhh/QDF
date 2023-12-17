from qiskit_aer import AerSimulator
from qiskit import transpile
from runner_help_function import *


def evaluate_circ(circuit, backend, shots):
    if backend == "GPU":
        sim = AerSimulator(method='statevector', device='GPU', cuStateVec_enable=True)
        tqc = transpile(circuit, sim)
        tqc.measure_all()
        res = sim.run(tqc, shots=shots)
        return res
    elif backend == "QPU":
        """
        这能跑出来个勾八个结果
        """
        return {}


def run_circuit(subcircuit_instances, subcircuit, instance_init_meas_ids, backend="QPU", shots=1024):
    """
    第一种方式：跑一个子电路的所有实例，其中evaluate_circ是可以开多进程或多线程的
    也就是其他所有部分都是单进程单线程，只有跑电路并行或并发。也就是一个子电路的所有电路结果出来后才可以获取进行接下来的操作
    :param backend:
    :param subcircuit_instances:
    :param subcircuit:
    :param instance_init_meas_ids:
    :param shots:
    :return:
    """
    run_circuit_res = {}
    for instance_init_meas in subcircuit_instances:
        if "Z" in instance_init_meas[1]:
            continue
        subcircuit_instance = modify_subcircuit_instance(
            subcircuit=subcircuit,
            init=instance_init_meas[0],
            meas=instance_init_meas[1],
        )
        # 这里可以用remote报起来，也可以多线程包起来
        subcircuit_inst_prob = evaluate_circ(
            circuit=subcircuit_instance, backend=backend, shots=shots
        )
        mutated_meas = mutate_measurement_basis(meas=instance_init_meas[1])
        for meas in mutated_meas:
            measured_prob = measure_prob(
                unmeasured_prob=subcircuit_inst_prob, meas=meas
            )
            instance_init_meas_id = instance_init_meas_ids[
                (instance_init_meas[0], meas)
            ]
            run_circuit_res[instance_init_meas_id] = measured_prob
    return run_circuit_res


def run_circuit_multi(instance_init_meas, subcircuit, instance_init_meas_ids, backend="QPU", shots=1024):
    """
    第二种方式：对每个实例都开多进程或多线程，返回的是和 run_circuit一样的结果类型，所有run_circuit_multi 并起来就是run_circuit的结果
    如果测量基是z基，返回的是空字典，记得剔除；I基返回的包含Z基的结果
    :param instance_init_meas:
    :param backend:
    :param subcircuit:
    :param instance_init_meas_ids:
    :param shots:
    :return:
    """
    res = {}
    if "Z" in instance_init_meas[1]:
        return {}
    subcircuit_instance = modify_subcircuit_instance(
        subcircuit=subcircuit,
        init=instance_init_meas[0],
        meas=instance_init_meas[1],
    )
    subcircuit_inst_prob = evaluate_circ(
        circuit=subcircuit_instance, backend=backend, shots=shots
    )
    mutated_meas = mutate_measurement_basis(meas=instance_init_meas[1])
    for meas in mutated_meas:
        measured_prob = measure_prob(
            unmeasured_prob=subcircuit_inst_prob, meas=meas
        )
        instance_init_meas_id = instance_init_meas_ids[
            (instance_init_meas[0], meas)
        ]
        res[instance_init_meas_id] = measured_prob
    return res
