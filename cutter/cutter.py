from utils import *
from cutter_help_functions import *
import gurobipy as gp
import math


class CutterModel(ABC):
    """
    可以当成抽象类，被继承，也可以直接写找切分点的求解器(把ABC，抽象方法给删掉)
    :return:
    """

    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        """
        这个用来返回结果
        :return:
        """
        pass


class MIP_Model(CutterModel):
    def __init__(
            self,
            n_vertices,
            edges,
            vertex_ids,
            id_vertices,
            num_subcircuit,
            max_subcircuit_width,
            max_subcircuit_cuts,
            subcircuit_size_imbalance,
            num_qubits,
            max_cuts,
    ):
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.vertex_ids = vertex_ids
        self.id_vertices = id_vertices
        self.num_subcircuit = num_subcircuit
        self.max_subcircuit_width = max_subcircuit_width
        self.max_subcircuit_cuts = max_subcircuit_cuts
        self.subcircuit_size_imbalance = math.sqrt(subcircuit_size_imbalance)
        self.num_qubits = num_qubits
        self.max_cuts = max_cuts

        """
        Count the number of input qubits directly connected to each node
        """
        self.vertex_weight = {}
        for node in self.vertex_ids:
            qargs = node.split(" ")
            num_in_qubits = 0
            for qarg in qargs:
                if int(qarg.split("]")[1]) == 0:
                    num_in_qubits += 1
            self.vertex_weight[node] = num_in_qubits

        self.model = gp.Model(name="cut_searching")
        self.model.params.OutputFlag = 0
        self._add_variables()
        self._add_constraints()

    def _add_variables(self):
        """
        Indicate if a vertex is in some subcircuit
        """
        self.vertex_var = []
        for i in range(self.num_subcircuit):
            subcircuit_y = []
            for j in range(self.n_vertices):
                j_in_i = self.model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY)
                subcircuit_y.append(j_in_i)
            self.vertex_var.append(subcircuit_y)

        """
        Indicate if an edge has one and only one vertex in some subcircuit
        """
        self.edge_var = []
        for i in range(self.num_subcircuit):
            subcircuit_x = []
            for j in range(self.n_edges):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=gp.GRB.BINARY)
                subcircuit_x.append(v)
            self.edge_var.append(subcircuit_x)

        """
        Total number of cuts
        add 0.1 for numerical stability
        """
        self.num_cuts = self.model.addVar(
            lb=0, ub=self.max_cuts + 0.1, vtype=gp.GRB.INTEGER, name="num_cuts"
        )

        self.subcircuit_counter = {}
        for subcircuit in range(self.num_subcircuit):
            self.subcircuit_counter[subcircuit] = {}

            self.subcircuit_counter[subcircuit]["original_input"] = self.model.addVar(
                lb=0.1,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="original_input_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["rho"] = self.model.addVar(
                lb=0,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="rho_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["O"] = self.model.addVar(
                lb=0,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="O_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["d"] = self.model.addVar(
                lb=0.1,
                ub=self.max_subcircuit_width,
                vtype=gp.GRB.INTEGER,
                name="d_%d" % subcircuit,
            )
            self.subcircuit_counter[subcircuit]["size"] = self.model.addVar(
                lb=self.n_vertices
                   / self.num_subcircuit
                   / self.subcircuit_size_imbalance,
                ub=self.n_vertices
                   / self.num_subcircuit
                   * self.subcircuit_size_imbalance,
                vtype=gp.GRB.INTEGER,
                name="size_%d" % subcircuit,
            )
            if self.max_subcircuit_cuts is not None:
                self.subcircuit_counter[subcircuit]["num_cuts"] = self.model.addVar(
                    lb=0.1,
                    ub=self.max_subcircuit_cuts,
                    vtype=gp.GRB.INTEGER,
                    name="num_cuts_%d" % subcircuit,
                )
        self.model.update()

    def _add_constraints(self):
        """
        each vertex in exactly one subcircuit
        """
        for v in range(self.n_vertices):
            self.model.addConstr(
                gp.quicksum(
                    [self.vertex_var[i][v] for i in range(self.num_subcircuit)]
                ),
                gp.GRB.EQUAL,
                1,
            )

        """
        edge_var=1 indicates one and only one vertex of an edge is in subcircuit
        edge_var[subcircuit][edge] = vertex_var[subcircuit][u] XOR vertex_var[subcircuit][v]
        """
        for i in range(self.num_subcircuit):
            for e in range(self.n_edges):
                u, v = self.edges[e]
                u_vertex_var = self.vertex_var[i][u]
                v_vertex_var = self.vertex_var[i][v]
                self.model.addConstr(self.edge_var[i][e] <= u_vertex_var + v_vertex_var)
                self.model.addConstr(self.edge_var[i][e] >= u_vertex_var - v_vertex_var)
                self.model.addConstr(self.edge_var[i][e] >= v_vertex_var - u_vertex_var)
                self.model.addConstr(
                    self.edge_var[i][e] <= 2 - u_vertex_var - v_vertex_var
                )

        """
        Symmetry-breaking constraints
        Force small-numbered vertices into small-numbered subcircuits:
            v0: in subcircuit 0
            v1: in subcircuit_0 or subcircuit_1
            v2: in subcircuit_0 or subcircuit_1 or subcircuit_2
            ...
        """
        for vertex in range(self.num_subcircuit):
            self.model.addConstr(
                gp.quicksum(
                    [
                        self.vertex_var[subcircuit][vertex]
                        for subcircuit in range(vertex + 1)
                    ]
                )
                == 1
            )

        """
        Compute number of cuts
        """
        self.model.addConstr(
            self.num_cuts
            == gp.quicksum(
                [
                    self.edge_var[subcircuit][i]
                    for i in range(self.n_edges)
                    for subcircuit in range(self.num_subcircuit)
                ]
            )
            / 2
        )

        for subcircuit in range(self.num_subcircuit):
            """
            Compute number of different types of qubit in a subcircuit
            """
            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["original_input"]
                == gp.quicksum(
                    [
                        self.vertex_weight[self.id_vertices[i]]
                        * self.vertex_var[subcircuit][i]
                        for i in range(self.n_vertices)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["rho"]
                == gp.quicksum(
                    [
                        self.edge_var[subcircuit][i]
                        * self.vertex_var[subcircuit][self.edges[i][1]]
                        for i in range(self.n_edges)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["O"]
                == gp.quicksum(
                    [
                        self.edge_var[subcircuit][i]
                        * self.vertex_var[subcircuit][self.edges[i][0]]
                        for i in range(self.n_edges)
                    ]
                )
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["d"]
                == self.subcircuit_counter[subcircuit]["original_input"]
                + self.subcircuit_counter[subcircuit]["rho"]
            )

            self.model.addConstr(
                self.subcircuit_counter[subcircuit]["size"]
                == gp.quicksum(
                    [self.vertex_var[subcircuit][v] for v in range(self.n_vertices)]
                )
            )

            if self.max_subcircuit_cuts is not None:
                self.model.addConstr(
                    self.subcircuit_counter[subcircuit]["num_cuts"]
                    == self.subcircuit_counter[subcircuit]["rho"]
                    + self.subcircuit_counter[subcircuit]["O"]
                )

        self.model.setObjective(self.num_cuts, gp.GRB.MINIMIZE)
        self.model.update()

    def check_graph(self, n_vertices, edges):
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        assert vertices == set(range(n_vertices))
        for u, v in edges:
            assert u < v
            assert u < n_vertices

    def solve(self):
        # print('solving for %d subcircuits'%self.num_subcircuit)
        # print('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        # % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
        try:
            self.model.params.threads = 48
            self.model.Params.TimeLimit = 30
            self.model.optimize()
        except (gp.GurobiError, AttributeError, Exception) as e:
            print("Caught: " + e.message)

        if self.model.solcount > 0:
            self.objective = None
            self.subcircuits = []
            self.optimal = self.model.Status == gp.GRB.OPTIMAL
            self.runtime = self.model.Runtime
            self.node_count = self.model.nodecount
            self.mip_gap = self.model.mipgap
            self.objective = self.model.ObjVal

            for i in range(self.num_subcircuit):
                subcircuit = []
                for j in range(self.n_vertices):
                    if abs(self.vertex_var[i][j].x) > 1e-4:
                        subcircuit.append(self.id_vertices[j])
                self.subcircuits.append(subcircuit)
            assert (
                    sum([len(subcircuit) for subcircuit in self.subcircuits])
                    == self.n_vertices
            )

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.num_subcircuit):
                for j in range(self.n_edges):
                    if abs(self.edge_var[i][j].x) > 1e-4 and j not in cut_edges_idx:
                        cut_edges_idx.append(j)
                        u, v = self.edges[j]
                        cut_edges.append((self.id_vertices[u], self.id_vertices[v]))
            self.cut_edges = cut_edges
            return True
        else:
            # print('Infeasible')
            return False


def find_cuts(qc: QuantumCircuit, cut_method='gurobi', cut_params=None, *args, **kwargs):
    """
    找可切分点
    :return:
    """
    verbose = kwargs['verbose']

    max_cuts = cut_params['max_cuts']
    num_subcircuits = cut_params['num_subcircuits']
    max_subcircuit_width = cut_params['max_subcircuit_width']
    max_subcircuit_cuts = cut_params['max_subcircuit_cuts']
    subcircuit_size_imbalance = cut_params['subcircuit_size_imbalance']

    stripped_circ = circuit_stripping(circuit=qc)
    n_vertices, edges, vertex_ids, id_vertices = read_circ(circuit=stripped_circ)
    num_qubits = qc.num_qubits

    for num_subcircuit in num_subcircuits:
        if (
                num_subcircuit * max_subcircuit_width - (num_subcircuit - 1) < num_qubits
                or num_subcircuit > num_qubits
                or max_cuts + 1 < num_subcircuit
        ):
            if verbose:
                print("%d subcircuits : IMPOSSIBLE" % (num_subcircuit))
            continue
        kwargs = dict(
            n_vertices=n_vertices,
            edges=edges,
            vertex_ids=vertex_ids,
            id_vertices=id_vertices,
            num_subcircuit=num_subcircuit,
            max_subcircuit_width=max_subcircuit_width,
            max_subcircuit_cuts=max_subcircuit_cuts,
            subcircuit_size_imbalance=subcircuit_size_imbalance,
            num_qubits=num_qubits,
            max_cuts=max_cuts,
        )

        mip_model = MIP_Model(**kwargs)
        feasible = mip_model.solve()
        if feasible:
            return mip_model
        elif verbose:
            print("%d subcircuits : NO SOLUTIONS" % (num_subcircuit))
    return None


def cut_parser(qc: QuantumCircuit, mip_model: MIP_Model, *args, **kwargs):
    """
    根据position,返回最终子电路信息以及连接信息等
    :return:
    """
    positions = cuts_parser(mip_model.cut_edges, qc)
    subcircuits, complete_path_map = subcircuits_parser(
        subcircuit_gates=mip_model.subcircuits, circuit=qc
    )
    O_rho_pairs = get_pairs(complete_path_map=complete_path_map)
    counter = get_counter(subcircuits=subcircuits, O_rho_pairs=O_rho_pairs)

    cut_solution = {
        "subcircuits": subcircuits,
        "complete_path_map": complete_path_map,
        "num_cuts": len(positions),
        "counter": counter,
    }
    return cut_solution


def cut(qc: QuantumCircuit,
        cut_params=None,
        cut_method="auto",
        verbose=False,
        *args, **kwargs):
    """
    cut_params:
        max_subcircuit_width,
        max_cuts,
        num_subcircuits,
        max_subcircuit_cuts,
        subcircuit_size_imbalance,

    用户最终调用的cut函数，输入的除了量子电路。如果输入了cut_solution，则不调用find_cuts;
    如果没有输入cut_solution，则走默认流程，此时还可以选择cut_method。
    :param verbose:
    :param cut_params:
    :param cut_solution:
    :param qc:
    :param cut_method:
    :param args:
    :param kwargs:
    :return:
    """
    position = find_cuts(qc, cut_params=cut_params, cut_method=cut_method, verbose=verbose)
    cut_solution = cut_parser(qc, position, verbose=verbose)
    return cut_solution
