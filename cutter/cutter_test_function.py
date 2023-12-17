from cutter_help_functions import *


def cut_parser_str_test(qc: QuantumCircuit, subcircuits, num_cuts, *args, **kwargs):
    """
    subcircuits 是cutqc的那种字符串表示，num_cuts就是切割了几下

    eg: mip_subcircuits = [["q[0]0 q[1]0", "q[1]1 q[2]0"], ["q[2]1 q[3]0", "q[3]1 q[4]0"]]
        mip_subcircuits = [["q[0]0 q[1]0", "q[1]1 q[2]0"], ["q[2]1 q[3]0", "q[2]2 q[4]0", "q[2]3 q[1]2", "q[2]4 q[0]1", "q[0]2 q[4]1"]]
    :return:
    """
    subcircuits, complete_path_map = subcircuits_parser(
        subcircuit_gates=subcircuits, circuit=qc
    )
    O_rho_pairs = get_pairs(complete_path_map=complete_path_map)
    counter = get_counter(subcircuits=subcircuits, O_rho_pairs=O_rho_pairs)

    cut_solution = {
        "subcircuits": subcircuits,
        "complete_path_map": complete_path_map,
        "num_cuts": num_cuts,
        "counter": counter,
    }
    return cut_solution
