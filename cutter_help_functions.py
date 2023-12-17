from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGOpNode, DAGCircuit
from qiskit import QuantumRegister, QuantumCircuit


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

def print_cutter_result(num_cuts, subcircuits, counter):
    print("Cutter result:")
    print("%d subcircuits, %d cuts" % (len(subcircuits), num_cuts))

    for subcircuit_idx in range(len(subcircuits)):
        print("subcircuit %d" % subcircuit_idx)
        print(
            "\u03C1 qubits = %d, O qubits = %d, width = %d, effective = %d, depth = %d, size = %d"
            % (
                counter[subcircuit_idx]["rho"],
                counter[subcircuit_idx]["O"],
                counter[subcircuit_idx]["d"],
                counter[subcircuit_idx]["effective"],
                counter[subcircuit_idx]["depth"],
                counter[subcircuit_idx]["size"],
            )
        )
        print(subcircuits[subcircuit_idx])


def cuts_parser(cuts, circ):
    dag = circuit_to_dag(circ)
    positions = []
    for position in cuts:
        source, dest = position
        source_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in source.split(" ")
        ]
        dest_qargs = [
            (x.split("]")[0] + "]", int(x.split("]")[1])) for x in dest.split(" ")
        ]
        qubit_cut = []
        # 这里并不能保证获取对应的position，cz可能连续作了两个
        for source_qarg in source_qargs:
            source_qubit, source_multi_Q_gate_idx = source_qarg
            for dest_qarg in dest_qargs:
                dest_qubit, dest_multi_Q_gate_idx = dest_qarg
                if (
                    source_qubit == dest_qubit
                    and dest_multi_Q_gate_idx == source_multi_Q_gate_idx + 1
                ):
                    qubit_cut.append(source_qubit)
        # 默认出现连续的相同双门，默认第一个
        for x in source.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                source_idx = int(x.split("]")[1])
        for x in dest.split(" "):
            if x.split("]")[0] + "]" == qubit_cut[0]:
                dest_idx = int(x.split("]")[1])
        multi_Q_gate_idx = max(source_idx, dest_idx)
        # 这一部分有什么必要吗，寄存器名还有可能不一样是吧，还真有可能
        wire = None
        for qubit in circ.qubits:
            if qubit.register.name == qubit_cut[0].split("[")[0] and qubit.index == int(
                qubit_cut[0].split("[")[1].split("]")[0]
            ):
                wire = qubit
        tmp = 0
        all_Q_gate_idx = None
        for gate_idx, gate in enumerate(
            list(dag.nodes_on_wire(wire=wire, only_ops=True))
        ):
            if len(gate.qargs) > 1:
                tmp += 1
                if tmp == multi_Q_gate_idx:
                    all_Q_gate_idx = gate_idx
        positions.append((wire, all_Q_gate_idx))
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions

def subcircuits_parser(subcircuit_gates, circuit):
    """
    Assign the single qubit gates to the closest two-qubit gates

    Returns:
    complete_path_map[input circuit qubit] = [{subcircuit_idx,subcircuit_qubit}]
    """

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
                # print('%s gate %d --> %s gate %d'%(qubit_A,qgate_A,qubit_B,qgate_B))
                if qubit_A == qubit_B:
                    distance = min(distance, abs(qgate_B - qgate_A))
        # print('Distance from %s to %s = %f'%(gate_A,gate_B,distance))
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
        gate_depth_encoding = gate_depth_encoding[:-1]
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
            # print('gate_depth_encoding = %s, MIP_gate_depth_encoding = %s'%(gate_depth_encoding,MIP_gate_depth_encoding))
            for subcircuit_idx in range(len(subcircuit_gates)):
                for gate_idx in range(len(subcircuit_gates[subcircuit_idx])):
                    if (
                        subcircuit_gates[subcircuit_idx][gate_idx]
                        == MIP_gate_depth_encoding
                    ):
                        subcircuit_gates[subcircuit_idx][gate_idx] = gate_depth_encoding
                        break
    # print('After translation :',subcircuit_gates,flush=True)
    subcircuit_op_nodes = {x: [] for x in range(len(subcircuit_gates))}
    subcircuit_sizes = [0 for x in range(len(subcircuit_gates))]
    ############################################################################################# 
    # 以上为把单门分配回去
    # 以下为生成complete path map
    complete_path_map = {}
    for circuit_qubit in dag.qubits:
        complete_path_map[circuit_qubit] = []
        qubit_ops = dag.nodes_on_wire(wire=circuit_qubit, only_ops=True)
        for qubit_op_idx, qubit_op in enumerate(qubit_ops):
            gate_depth_encoding = gate_depth_encodings[qubit_op]
            nearest_subcircuit_idx = -1
            min_distance = float("inf")
            for subcircuit_idx in range(len(subcircuit_gates)):
                distance = float("inf")
                for gate in subcircuit_gates[subcircuit_idx]:
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
                # print('{} op #{:d} {:s} encoding = {:s}'.format(circuit_qubit,qubit_op_idx,qubit_op.name,gate_depth_encoding),
                # 'belongs in subcircuit %d'%nearest_subcircuit_idx)
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