from utils import *


def build(results, pos):
    """
    根据results和对应pos去进行重建
    """


def pre_build(subcircuit_entry, instance_init_meas_ids, entry_init_meas_ids, run_circuit_res):
    """
    对每个子电路的所有实体进行预处理
    """
    attribute_shots_res = {}
    attribute_shots_jobs = list(subcircuit_entry.keys())
    attribute_shots_jobs = {
        key: subcircuit_entry[key] for key in attribute_shots_jobs
    }
    for subcircuit_entry_init_meas in attribute_shots_jobs:
        subcircuit_entry_term = attribute_shots_jobs[subcircuit_entry_init_meas]
        subcircuit_entry_prob = None
        for term in subcircuit_entry_term:
            coefficient, subcircuit_instance_init_meas = term
            subcircuit_instance_init_meas_id = instance_init_meas_ids[subcircuit_instance_init_meas]
            if subcircuit_entry_prob is None:
                subcircuit_entry_prob = coefficient * run_circuit_res[subcircuit_instance_init_meas_id]
            else:
                subcircuit_entry_prob += coefficient * run_circuit_res[subcircuit_instance_init_meas_id]
        entry_init_meas_id = entry_init_meas_ids[subcircuit_entry_init_meas]
        attribute_shots_res[entry_init_meas_id] = subcircuit_entry_prob
    return attribute_shots_res


def pre_build_multi(subcircuit_entry_init_meas, subcircuit_entry_term, instance_init_meas_ids, entry_init_meas_ids, run_circuit_res):
    """
    每个线程或进程对应一个entry
    """
    attribute_shots_res = {}
    subcircuit_entry_prob = None
    for term in subcircuit_entry_term:
        coefficient, subcircuit_instance_init_meas = term
        subcircuit_instance_init_meas_id = instance_init_meas_ids[subcircuit_instance_init_meas]
        if subcircuit_entry_prob is None:
            subcircuit_entry_prob = coefficient * run_circuit_res[subcircuit_instance_init_meas_id]
        else:
            subcircuit_entry_prob += coefficient * run_circuit_res[subcircuit_instance_init_meas_id]
    entry_init_meas_id = entry_init_meas_ids[subcircuit_entry_init_meas]
    attribute_shots_res[entry_init_meas_id] = subcircuit_entry_prob
    return attribute_shots_res
