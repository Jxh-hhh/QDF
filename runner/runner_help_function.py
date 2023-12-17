import itertools

import numpy as np
from qiskit.circuit.library import XGate, HGate, SGate, SdgGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
import copy


def modify_subcircuit_instance(subcircuit, init, meas):
    """
    Modify the different init, meas for a given subcircuit
    Returns:
    Modified subcircuit_instance
    List of mutated measurements
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)
    return subcircuit_instance_circuit


def mutate_measurement_basis(meas):
    """
    I and Z measurement basis correspond to the same logical circuit
    """
    if all(x != "I" for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != "I":
                mutated_meas.append([x])
            else:
                mutated_meas.append(["I", "Z"])
        mutated_meas = list(itertools.product(*mutated_meas))
        return mutated_meas


def measure_prob(unmeasured_prob, meas):
    if meas.count("comp") == len(meas) or type(unmeasured_prob) is float:
        return unmeasured_prob
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        # print('Measuring in',meas)
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            measured_prob[effective_state] += sigma * p
        return measured_prob


def measure_state(full_state, meas):
    """
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1]):
        if meas_bit == "1" and meas_basis != "I" and meas_basis != "comp":
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))
    return sigma, effective_state
