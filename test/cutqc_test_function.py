from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate
import copy
from qiskit.quantum_info import Statevector
from qiskit.providers import aer
import itertools

def circuit_stripping(circuit):
    # Remove all single qubit gates and barriers in the circuit
    dag = circuit_to_dag(circuit)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circuit.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) == 2 and vertex.op.name != "barrier":
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)
def read_circ(circuit):
    dag = circuit_to_dag(circuit)
    edges = []
    node_name_ids = {}
    id_node_names = {}
    vertex_ids = {}
    curr_node_id = 0
    qubit_gate_counter = {}
    for qubit in dag.qubits:
        qubit_gate_counter[qubit] = 0
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception("vertex does not have 2 qargs!")
        arg0, arg1 = vertex.qargs
        vertex_name = "%s[%d]%d %s[%d]%d" % (
            arg0.register.name,
            arg0.index,
            qubit_gate_counter[arg0],
            arg1.register.name,
            arg1.index,
            qubit_gate_counter[arg1],
        )
        qubit_gate_counter[arg0] += 1
        qubit_gate_counter[arg1] += 1
        # print(vertex.op.label,vertex_name,curr_node_id)
        if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            vertex_ids[id(vertex)] = curr_node_id
            curr_node_id += 1

    for u, v, _ in dag.edges():
        if isinstance(u, DAGOpNode) and isinstance(v, DAGOpNode):
            u_id = vertex_ids[id(u)]
            v_id = vertex_ids[id(v)]
            edges.append((u_id, v_id))

    n_vertices = dag.size()
    return n_vertices, edges, node_name_ids, id_node_names

def subcircuits_parser(subcircuit_gates, circuit):
    """
    Assign the single qubit gates to the closest two-qubit gates

    Returns:
    complete_path_map[input circuit qubit] = [{subcircuit_idx,subcircuit_qubit}]
    """
    # 两个门在同个比特上的距离
    def calculate_distance_between_gate(gate_A, gate_B):
        if len(gate_A.split(" ")) >= len(gate_B.split(" ")):
            tmp_gate = gate_A
            gate_A = gate_B
            gate_B = tmp_gate
        distance = float("inf")
        for qarg_A in gate_A.split(" "):
            qubit_A = qarg_A.split("]")[0] + "]"
            qgate_A = int(qarg_A.split("]")[-1])
            for qarg_B in gate_B.split(" "):
                qubit_B = qarg_B.split("]")[0] + "]"
                qgate_B = int(qarg_B.split("]")[-1])
                if qubit_A == qubit_B:
                    distance = min(distance, abs(qgate_B - qgate_A))
        return distance

    dag = circuit_to_dag(circuit)
    qubit_allGate_depths = {x: 0 for x in circuit.qubits}
    qubit_2qGate_depths = {x: 0 for x in circuit.qubits}
    gate_depth_encodings = {}
    # print('Before translation :',subcircuit_gates,flush=True)
    for op_node in dag.topological_op_nodes():
        gate_depth_encoding = ""
        for qarg in op_node.qargs:
            gate_depth_encoding += "%s[%d]%d " % (
                qarg.register.name,
                qarg.index,
                qubit_allGate_depths[qarg],
            )

        # 在这里 gate_depth_encoding 还没看出来有啥用，只是填了qubit信息和深度，存到字典里
        gate_depth_encoding = gate_depth_encoding[:-1]  # 最后去掉空格
        gate_depth_encodings[op_node] = gate_depth_encoding
        for qarg in op_node.qargs:
            qubit_allGate_depths[qarg] += 1
        if len(op_node.qargs) == 2:
            MIP_gate_depth_encoding = ""
            for qarg in op_node.qargs:
                MIP_gate_depth_encoding += "%s[%d]%d " % (
                    qarg.register.name,
                    qarg.index,
                    qubit_2qGate_depths[qarg],
                )
                qubit_2qGate_depths[qarg] += 1
            MIP_gate_depth_encoding = MIP_gate_depth_encoding[:-1]
            # 去原来的里边找这个门呗，把这个结构重返有单门的时候
            for subcircuit_idx in range(len(subcircuit_gates)):
                for gate_idx in range(len(subcircuit_gates[subcircuit_idx])):
                    if (
                        subcircuit_gates[subcircuit_idx][gate_idx]
                        == MIP_gate_depth_encoding
                    ):
                        subcircuit_gates[subcircuit_idx][gate_idx] = gate_depth_encoding
                        break

    subcircuit_op_nodes = {x: [] for x in range(len(subcircuit_gates))}
    subcircuit_sizes = [0 for x in range(len(subcircuit_gates))]
    complete_path_map = {}
    # 最重要是这个complete path map 是什么东西
    for circuit_qubit in dag.qubits:
        complete_path_map[circuit_qubit] = []
        qubit_ops = dag.nodes_on_wire(wire=circuit_qubit, only_ops=True)
        for qubit_op_idx, qubit_op in enumerate(qubit_ops): # 对每个qubit的op节点进行遍历
            gate_depth_encoding = gate_depth_encodings[qubit_op]
            # 这个最近的电路下标到底是什么鬼
            nearest_subcircuit_idx = -1
            min_distance = float("inf")
            for subcircuit_idx in range(len(subcircuit_gates)):
                distance = float("inf")
                for gate in subcircuit_gates[subcircuit_idx]:
                    # 还把单门给略过了
                    if len(gate.split(" ")) == 1:
                        # Do not compare against single qubit gates
                        continue
                    else:
                        distance = min(
                            distance,
                            calculate_distance_between_gate(
                                gate_A=gate_depth_encoding, gate_B=gate
                            ),
                        )
                # print('Distance from %s to subcircuit %d = %f'%(gate_depth_encoding,subcircuit_idx,distance))
                if distance < min_distance:
                    min_distance = distance
                    nearest_subcircuit_idx = subcircuit_idx
            assert nearest_subcircuit_idx != -1
            path_element = {
                "subcircuit_idx": nearest_subcircuit_idx,
                "subcircuit_qubit": subcircuit_sizes[nearest_subcircuit_idx],
            }
            if (
                len(complete_path_map[circuit_qubit]) == 0
                or nearest_subcircuit_idx
                != complete_path_map[circuit_qubit][-1]["subcircuit_idx"]
            ):
                complete_path_map[circuit_qubit].append(path_element)
                subcircuit_sizes[nearest_subcircuit_idx] += 1

            subcircuit_op_nodes[nearest_subcircuit_idx].append(qubit_op)
    for circuit_qubit in complete_path_map:
        # print(circuit_qubit,'-->')
        for path_element in complete_path_map[circuit_qubit]:
            path_element_qubit = QuantumRegister(
                size=subcircuit_sizes[path_element["subcircuit_idx"]], name="q"
            )[path_element["subcircuit_qubit"]]
            path_element["subcircuit_qubit"] = path_element_qubit
            # print(path_element)
    subcircuits = generate_subcircuits(
        subcircuit_op_nodes=subcircuit_op_nodes,
        complete_path_map=complete_path_map,
        subcircuit_sizes=subcircuit_sizes,
        dag=dag,
    )
    return subcircuits, complete_path_map
    # return complete_path_map
def generate_subcircuits(subcircuit_op_nodes, complete_path_map, subcircuit_sizes, dag):
    qubit_pointers = {x: 0 for x in complete_path_map}
    subcircuits = [QuantumCircuit(x, name="q") for x in subcircuit_sizes]
    for op_node in dag.topological_op_nodes():
        subcircuit_idx = list(
            filter(
                lambda x: op_node in subcircuit_op_nodes[x], subcircuit_op_nodes.keys()
            )
        )
        assert len(subcircuit_idx) == 1
        subcircuit_idx = subcircuit_idx[0]
        # print('{} belongs in subcircuit {:d}'.format(op_node.qargs,subcircuit_idx))
        subcircuit_qargs = []
        for op_node_qarg in op_node.qargs:
            if (
                complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]][
                    "subcircuit_idx"
                ]
                != subcircuit_idx
            ):
                qubit_pointers[op_node_qarg] += 1
            path_element = complete_path_map[op_node_qarg][qubit_pointers[op_node_qarg]]
            assert path_element["subcircuit_idx"] == subcircuit_idx
            subcircuit_qargs.append(path_element["subcircuit_qubit"])
        # print('-->',subcircuit_qargs)
        subcircuits[subcircuit_idx].append(
            instruction=op_node.op, qargs=subcircuit_qargs, cargs=None
        )
    return subcircuits
def get_pairs(complete_path_map):
    O_rho_pairs = []
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        if len(path) > 1:
            for path_ctr, item in enumerate(path[:-1]):
                O_qubit_tuple = item
                rho_qubit_tuple = path[path_ctr + 1]
                O_rho_pairs.append((O_qubit_tuple, rho_qubit_tuple))
    return O_rho_pairs

def get_counter(subcircuits, O_rho_pairs):
    counter = {}
    for subcircuit_idx, subcircuit in enumerate(subcircuits):
        counter[subcircuit_idx] = {
            "effective": subcircuit.num_qubits,
            "rho": 0,
            "O": 0,
            "d": subcircuit.num_qubits,
            "depth": subcircuit.depth(),
            "size": subcircuit.size(),
        }
    for pair in O_rho_pairs:
        O_qubit, rho_qubit = pair
        counter[O_qubit["subcircuit_idx"]]["effective"] -= 1
        counter[O_qubit["subcircuit_idx"]]["O"] += 1
        counter[rho_qubit["subcircuit_idx"]]["rho"] += 1
    return counter
class ComputeGraph(object):
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, subcircuit_idx, attributes):
        self.nodes[subcircuit_idx] = attributes

    def remove_node(self, subcircuit_idx):
        """
        Remove a node from the compute graph
        """
        del self.nodes[subcircuit_idx]


    def add_edge(self, u_for_edge, v_for_edge, attributes):
        self.edges.append((u_for_edge, v_for_edge, attributes))

    def get_edges(self, from_node, to_node):
        """
        Get edges in the graph based on some given conditions:
        1. If from_node is given. Only retain edges from the node.
        2. If to_node is given. Only retain edges to the node.
        """
        edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            match_from_node = from_node is None or u_for_edge == from_node
            match_to_node = to_node is None or v_for_edge == to_node
            if match_from_node and match_to_node:
                edges.append(edge)
        return edges

    def assign_bases_to_edges(self, edge_bases, edges):
        """Assign the edge_bases to edges"""
        for edge_basis, edge in zip(edge_bases, edges):
            assert edge in self.edges
            u_for_edge, v_for_edge, attributes = edge
            attributes["basis"] = edge_basis

    def remove_bases_from_edges(self, edges):
        """Remove the edge_bases from edges"""
        for edge in edges:
            u_for_edge, v_for_edge, attributes = edge
            if "basis" in attributes:
                del attributes["basis"]

    def remove_all_bases(self):
        for edge in self.edges:
            u_for_edge, v_for_edge, attributes = edge
            if "basis" in attributes:
                del attributes["basis"]

    def get_init_meas(self, subcircuit_idx):
        """Get the entry_init, entry_meas for a given node"""
        node_attributes = self.nodes[subcircuit_idx]
        bare_subcircuit = node_attributes["subcircuit"]
        entry_init = ["zero"] * bare_subcircuit.num_qubits
        edges_to_node = self.get_edges(from_node=None, to_node=subcircuit_idx)
        for edge in edges_to_node:
            _, v_for_edge, edge_attributes = edge
            assert v_for_edge == subcircuit_idx
            entry_init[
                bare_subcircuit.qubits.index(edge_attributes["rho_qubit"])
            ] = edge_attributes["basis"]

        entry_meas = ["comp"] * bare_subcircuit.num_qubits
        edges_from_node = self.get_edges(from_node=subcircuit_idx, to_node=None)
        for edge in edges_from_node:
            u_for_edge, _, edge_attributes = edge
            assert u_for_edge == subcircuit_idx
            entry_meas[
                bare_subcircuit.qubits.index(edge_attributes["O_qubit"])
            ] = edge_attributes["basis"]
        return (tuple(entry_init), tuple(entry_meas))

    def get_contraction_edges(
        self, leading_subcircuit_idx, contracted_subcircuits_indices
    ):
        """
        Edges connecting the leading subcircuit and any one of the contracted subcircuits
        """
        contraction_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge == leading_subcircuit_idx
                and v_for_edge in contracted_subcircuits_indices
            ):
                contraction_edges.append(edge)
            elif (
                v_for_edge == leading_subcircuit_idx
                and u_for_edge in contracted_subcircuits_indices
            ):
                contraction_edges.append(edge)
        return contraction_edges

    def get_leading_edges(self, leading_subcircuit_idx, contracted_subcircuits_indices):
        """
        Edges only connecting the leading subcircuit
        """
        leading_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge == leading_subcircuit_idx
                and v_for_edge not in contracted_subcircuits_indices
            ):
                leading_edges.append(edge)
            elif (
                v_for_edge == leading_subcircuit_idx
                and u_for_edge not in contracted_subcircuits_indices
            ):
                leading_edges.append(edge)
        return leading_edges

    def get_trailing_edges(
        self, leading_subcircuit_idx, contracted_subcircuits_indices
    ):
        """
        Edges only connecting the contracted subcircuits
        """
        trailing_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                continue
            elif (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge != leading_subcircuit_idx
            ):
                trailing_edges.append(edge)
            elif (
                v_for_edge in contracted_subcircuits_indices
                and u_for_edge != leading_subcircuit_idx
            ):
                trailing_edges.append(edge)
        return trailing_edges

    def get_contracted_edges(self, contracted_subcircuits_indices):
        """
        Edges in between the contracted subcircuits
        """
        contracted_edges = []
        for edge in self.edges:
            u_for_edge, v_for_edge, _ = edge
            if (
                u_for_edge in contracted_subcircuits_indices
                and v_for_edge in contracted_subcircuits_indices
            ):
                contracted_edges.append(edge)
        return contracted_edges

def generate_compute_graph(counter, subcircuits, complete_path_map):
    """
    Generate the connection graph among subcircuits
    """
    compute_graph = ComputeGraph()
    for subcircuit_idx in counter:
        subcircuit_attributes = copy.deepcopy(counter[subcircuit_idx])
        subcircuit_attributes["subcircuit"] = subcircuits[subcircuit_idx]
        compute_graph.add_node(
            subcircuit_idx=subcircuit_idx, attributes=subcircuit_attributes
        )
    for circuit_qubit in complete_path_map:
        path = complete_path_map[circuit_qubit]
        for counter in range(len(path) - 1):
            upstream_subcircuit_idx = path[counter]["subcircuit_idx"]
            downstream_subcircuit_idx = path[counter + 1]["subcircuit_idx"]
            compute_graph.add_edge(
                u_for_edge=upstream_subcircuit_idx,
                v_for_edge=downstream_subcircuit_idx,
                attributes={
                    "O_qubit": path[counter]["subcircuit_qubit"],
                    "rho_qubit": path[counter + 1]["subcircuit_qubit"],
                },
            )
    return compute_graph
def get_instance_init_meas(init_label, meas_label):
    """
    Convert subcircuit entry init,meas into subcircuit instance init,meas
    """
    init_combinations = []
    for x in init_label:
        if x == "zero":
            init_combinations.append(["zero"])
        elif x == "I":
            init_combinations.append(["+zero", "+one"])
        elif x == "X":
            init_combinations.append(["2plus", "-zero", "-one"])
        elif x == "Y":
            init_combinations.append(["2plusI", "-zero", "-one"])
        elif x == "Z":
            init_combinations.append(["+zero", "-one"])
        else:
            raise Exception("Illegal initilization symbol :", x)
    init_combinations = list(itertools.product(*init_combinations))

    subcircuit_init_meas = []
    for init in init_combinations:
        subcircuit_init_meas.append((tuple(init), tuple(meas_label)))
    return subcircuit_init_meas


def convert_to_physical_init(init):
    coefficient = 1
    for idx, x in enumerate(init):
        if x == "zero":
            continue
        elif x == "+zero":
            init[idx] = "zero"
        elif x == "+one":
            init[idx] = "one"
        elif x == "2plus":
            init[idx] = "plus"
            coefficient *= 2
        elif x == "-zero":
            init[idx] = "zero"
            coefficient *= -1
        elif x == "-one":
            init[idx] = "one"
            coefficient *= -1
        elif x == "2plusI":
            init[idx] = "plusI"
            coefficient *= 2
        else:
            raise Exception("Illegal initilization symbol :", x)
    return coefficient, tuple(init)
def generate_subcircuit_entries(compute_graph):
    """
    subcircuit_entries[subcircuit_idx][entry_init, entry_meas] = subcircuit_entry_term
    subcircuit_entry_term (list): (coefficient, instance_init, instance_meas)
    subcircuit_entry = Sum(coefficient*subcircuit_instance)

    subcircuit_instances[subcircuit_idx] = [(instance_init,instance_meas)]
    """
    subcircuit_entries = {}
    subcircuit_instances = {}
    for subcircuit_idx in compute_graph.nodes:
        # print('subcircuit_%d'%subcircuit_idx)
        bare_subcircuit = compute_graph.nodes[subcircuit_idx]["subcircuit"]
        subcircuit_entries[subcircuit_idx] = {}
        subcircuit_instances[subcircuit_idx] = []
        from_edges = compute_graph.get_edges(from_node=subcircuit_idx, to_node=None)
        to_edges = compute_graph.get_edges(from_node=None, to_node=subcircuit_idx)
        subcircuit_edges = from_edges + to_edges
        for subcircuit_edge_bases in itertools.product(
            ["I", "X", "Y", "Z"], repeat=len(subcircuit_edges)
        ):
            # print('subcircuit_edge_bases =',subcircuit_edge_bases)
            subcircuit_entry_init = ["zero"] * bare_subcircuit.num_qubits
            subcircuit_entry_meas = ["comp"] * bare_subcircuit.num_qubits
            for edge_basis, edge in zip(subcircuit_edge_bases, subcircuit_edges):
                (
                    upstream_subcircuit_idx,
                    downstream_subcircuit_idx,
                    edge_attributes,
                ) = edge
                if subcircuit_idx == upstream_subcircuit_idx:
                    O_qubit = edge_attributes["O_qubit"]
                    subcircuit_entry_meas[
                        bare_subcircuit.qubits.index(O_qubit)
                    ] = edge_basis
                elif subcircuit_idx == downstream_subcircuit_idx:
                    rho_qubit = edge_attributes["rho_qubit"]
                    subcircuit_entry_init[
                        bare_subcircuit.qubits.index(rho_qubit)
                    ] = edge_basis
                else:
                    raise IndexError(
                        "Generating entries for a subcircuit. subcircuit_idx should be either upstream or downstream"
                    )
            # 以上给赋了个 meas init
            subcircuit_instance_init_meas = get_instance_init_meas(
                init_label=subcircuit_entry_init, meas_label=subcircuit_entry_meas
            )
            subcircuit_entry_term = []
            for init_meas in subcircuit_instance_init_meas:
                instance_init, instance_meas = init_meas
                coefficient, instance_init = convert_to_physical_init(
                    init=list(instance_init)
                )
                if (instance_init, instance_meas) not in subcircuit_instances[
                    subcircuit_idx
                ]:
                    subcircuit_instances[subcircuit_idx].append(
                        (instance_init, instance_meas)
                    )
                subcircuit_entry_term.append(
                    (coefficient, (instance_init, instance_meas))
                )
                # print('%d *'%coefficient, instance_init, instance_meas)
            subcircuit_entries[subcircuit_idx][
                (tuple(subcircuit_entry_init), tuple(subcircuit_entry_meas))
            ] = subcircuit_entry_term
    return subcircuit_entries, subcircuit_instances
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


def evaluate_circ(circuit, backend, options=None):
    circuit = copy.deepcopy(circuit)
    if backend == "statevector_simulator":
        simulator = aer.Aer.get_backend("statevector_simulator")
        result = simulator.run(circuit).result()
        statevector = result.get_statevector(circuit)
        prob_vector = Statevector(statevector).probabilities()
        return prob_vector
    elif backend == "noiseless_qasm_simulator":
        simulator = aer.Aer.get_backend("aer_simulator")
        if isinstance(options, dict) and "num_shots" in options:
            num_shots = options["num_shots"]
        else:
            num_shots = max(1024, 2**circuit.num_qubits)

        if isinstance(options, dict) and "memory" in options:
            memory = options["memory"]
        else:
            memory = False
        if circuit.num_clbits == 0:
            circuit.measure_all()
        result = simulator.run(circuit, shots=num_shots, memory=memory).result()

        if memory:
            qasm_memory = np.array(result.get_memory(circuit))
            assert len(qasm_memory) == num_shots
            return qasm_memory
        else:
            noiseless_counts = result.get_counts(circuit)
            assert sum(noiseless_counts.values()) == num_shots
            noiseless_counts = dict_to_array(
                distribution_dict=noiseless_counts, force_prob=True
            )
            return noiseless_counts
    else:
        raise NotImplementedError
def dict_to_array(distribution_dict, force_prob):
    state = list(distribution_dict.keys())[0]
    num_qubits = len(state)
    num_shots = sum(distribution_dict.values())
    cnts = np.zeros(2**num_qubits, dtype=float)
    for state in distribution_dict:
        cnts[int(state, 2)] = distribution_dict[state]
    if abs(sum(cnts) - num_shots) > 1:
        print(
            "dict_to_array may be wrong, converted counts = {}, input counts = {}".format(
                sum(cnts), num_shots
            )
        )
    if not force_prob:
        return cnts
    else:
        prob = cnts / num_shots
        assert abs(sum(prob) - 1) < 1e-10
        return prob
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

def merge_prob_vector(unmerged_prob_vector, qubit_states):
    num_active = qubit_states.count("active")
    num_merged = qubit_states.count("merged")
    merged_prob_vector = np.zeros(2**num_active, dtype="float32")
    # print('merging with qubit states {}. {:d}-->{:d}'.format(
    #     qubit_states,
    #     len(unmerged_prob_vector),len(merged_prob_vector)))
    for active_qubit_states in itertools.product(["0", "1"], repeat=num_active):
        if len(active_qubit_states) > 0:
            merged_bin_id = int("".join(active_qubit_states), 2)
        else:
            merged_bin_id = 0
        for merged_qubit_states in itertools.product(["0", "1"], repeat=num_merged):
            active_ptr = 0
            merged_ptr = 0
            binary_state_id = ""
            for qubit_state in qubit_states:
                if qubit_state == "active":
                    binary_state_id += active_qubit_states[active_ptr]
                    active_ptr += 1
                elif qubit_state == "merged":
                    binary_state_id += merged_qubit_states[merged_ptr]
                    merged_ptr += 1
                else:
                    binary_state_id += "%s" % qubit_state
            state_id = int(binary_state_id, 2)
            merged_prob_vector[merged_bin_id] += unmerged_prob_vector[state_id]
    return merged_prob_vector

def distribute_load(capacities):
    total_load = sum(capacities.values())
    total_capacity = sum(capacities.values())
    loads = {subcircuit_idx: 0 for subcircuit_idx in capacities}

    for slot_idx in loads:
        loads[slot_idx] = int(capacities[slot_idx] / total_capacity * total_load)
    total_load -= sum(loads.values())

    for slot_idx in loads:
        while total_load > 0 and loads[slot_idx] < capacities[slot_idx]:
            loads[slot_idx] += 1
            total_load -= 1
    # print('capacities = {}. total_capacity = {:d}'.format(capacities,total_capacity))
    # print('loads = {}. remaining total_load = {:d}'.format(loads,total_load))
    assert total_load == 0
    return loads