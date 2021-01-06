import random
import warnings
from datetime import datetime
from time import sleep

import networkx as nx
import numpy as np
import qiskit as qk
from qiskit.quantum_info.operators import Operator

from .graph_states import GraphDB, _plots, nx_plot_options

global seed
seed = 1234
np.random.seed(seed)

def partitions(n, start=2):
    """Return all possible partitions of integer n

    Arguments:
        n {int} -- integer

    Keyword Arguments:
        start {int} -- starting position (default: {2})

    Yields:
        tuple -- returns tuples of partitions
    """
    yield(n,)
    for i in range(start, n//2 + 1):
        for p in partitions(n-i, i):
             yield (i,) + p


class CircuitConstructor:
    """Parent class of constructing circuits

    Raises:
        NotImplementedError: build_circuit() must be overriden by child class
        NotImplementedError: estimate_entanglement() must be overriden by child class
    """
    def __init__(self,n_qubits= 2, n_shots=1024, verbose=False,
                 state_simulation=True, save_to_file=False):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.runtime = datetime.now().strftime("%Y_%m_%d")
        self.statevec = None
        self.circuit =  None
        self.state_sim = state_simulation
        self.counts = None

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def build_circuit(self):
        raise NotImplementedError

    def save_circuit(self):
        if self.save_to_file and self.circuit is not None:
            with open('circuit_'+self.runtime+'.txt','w') as fp:
                for line in self.circuit.qasm():
                    fp.write(line)
        return

    def execute_circuit(self):
        """execute circuit with state_vector sim
        """
        if self.state_sim:
            self.statevec=qk.execute(self.circuit,\
                                    backend=qk.Aer.get_backend('statevector_simulator'),
                                    shots=self.n_shots).result().get_statevector()
        else:
            self.counts = qk.execute(self.circuit,
                                    backend=qk.Aer.get_backend('qasm_simulator'),
                                    seed_simulator=seed, shots=self.n_shots).result().get_counts()
        return

    def estimate_entanglement(self):
        """estimate entanglement of final state
        using n-qubit entanglement winessess
        if circuit was prepared as GHZ state then assume maximally entangled
        if circuit was prepared as random graph state then use witnesses
        """
        raise NotImplementedError


class GraphCircuit(CircuitConstructor):
    """Class to construct circuits from graph states

    Arguments:
        CircuitConstructor {Parent class} -- abstract class
    """
    def __init__(self, graph_db=GraphDB(), gate_type="Controlled_Phase", smallest_subgraph=2,
                       largest_subgraph=None, **kwargs):
        super(GraphCircuit, self).__init__(**kwargs)
        if isinstance(graph_db, GraphDB):
            self.graph_db = graph_db
        self.state_vec = True
        self.gate_type = gate_type
        self.all_graphs = self.get_sorted_db()
        self.largest_subgraph = self.check_largest(largest_subgraph)
        self.smallest_subgraph = max(2, smallest_subgraph)
        self.graph_combs = self.generate_all_subgraphs()
        self.ops_labels = None

    def check_largest(self, val):
        for key, itm in self.all_graphs.items():
                if len(itm) != 0:
                    max_subgraph = key
        if val is None:
            return max_subgraph
        if val > max_subgraph:
            warnings.warn("The largest possible subgraph in the database has %s nodes" %max_subgraph)
            warnings.warn("Resetting largest possible subgraph: %s --> %s" %(val, max_subgraph))
            return max_subgraph
        return val

    def generate_all_subgraphs(self):
        combs = list(set(partitions(self.n_qubits, start=self.smallest_subgraph)))
        for (itm, comb) in enumerate(combs):
            if any([itm > self.largest_subgraph for itm in comb]):
                combs.pop(itm)
        if len(combs) == 0:
            raise ValueError("Empty list of subgraph combinations. Circuit cannot be constructed as specified.")
        return combs

    def combine_subgraphs(self, sub_graphs):
        union_graph = nx.DiGraph()
        for sub_g in sub_graphs:
            union_nodes = len(union_graph.nodes)
            if union_nodes > 1:
                first_node = random.randint(0, union_graph.order() - 1)
                offset = len(sub_g.nodes)
                second_nodes = np.random.randint(union_graph.order(), sub_g.order() + union_graph.order() - 1,sub_g.order())
                #second_node = random.randint(union_nodes, offset + union_nodes - 1)
                union_graph = nx.disjoint_union(union_graph, sub_g)
                #union_graph.add_weighted_edges_from([(first_node, second_node, 1.0)])
                for idx in second_nodes:
                    union_graph.add_weighted_edges_from([(first_node, idx, 1.0)])
            else:
                union_graph = nx.disjoint_union(union_graph, sub_g)
        return union_graph

    def pick_subgraphs(self):
        comb_idx = random.randint(0, len(self.graph_combs) - 1)
        comb = self.graph_combs[comb_idx]
        self.print_verbose("Configuration with {} Subgraphs with # nodes:{}".format(len(comb), comb))
        sub_graphs = []
        for num_nodes in comb:
            sub_g = self.all_graphs[num_nodes]
            idx = random.randint(0, len(sub_g) - 1)
            sub_graphs.append(sub_g[idx])
        return sub_graphs

    def build_circuit(self, graph_plot=False):
        # 1. Pick a random combination of subgraphs
        sub_graphs = self.pick_subgraphs()
        # 2. Combine subgraphs into a single circuit graph
        circuit_graph = self.combine_subgraphs(sub_graphs)
        self.circuit_graph = circuit_graph
        if graph_plot and _plots:
            nx.draw_circular(circuit_graph, **nx_plot_options)
        # 3. Build a circuit from the graph state
        if self.gate_type == "Controlled_Phase":
            self.print_verbose("Assigning a Controlled Phase Gate (H-CNOT-H) to Node Edges")
            self._build_controlled_phase_gate(circuit_graph)
        elif self.gate_type == "SControlled_Phase": # same as controlled phase gate but w/ Stochastic unitary gates after CNOT
            self.print_verbose("Assigning a Stochastic Controlled Phase Gate (H-CNOT-P(U)-H) to Node Edges")
            self._build_Scontrolled_phase_gate(circuit_graph)

    def get_generators(self):
        """ get generators of the graph state stabilizers
        generators are n-length strings (n = number of qubits)"""
        generators = []
        for idx in self.circuit_graph.nodes():
            temp = list('I'*self.circuit_graph.order())
            temp[vdx]='X'
            for jdx in self.circuit_graph.neighbors(idx):
                temp[jdx]='Z'
            temp = "".join(temp)
            generators.append(temp)
        self.generators = generators


    def get_stabilizers(self):
        """ get the stabilizer operators for a graph state """
        stabilizers = []
        binary_keys = [np.binary_repr(x,self.n_qubits) \
                                for x in range(2**self.n_qubits)]
        # 'IIII...' operator always included, this corresponds to binary key '000...'
        stabilizers.append('+I'*n_qubits)
        for idx in binary_keys:
            coefs = [int(x) for x in list(idx)]
            temp = []
            for jdx in coefs:
                if jdx==0:
                    temp.append('I')
                else:
                    pass
        self.stabilizers = stabilizers

    def _build_controlled_phase_gate(self, graph):
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        gate_pairs = []
        for node, ngbrs in graph.adjacency():
            for ngbr, _ in ngbrs.items():
                circ.h(node)
                circ.cx(node, ngbr)
                circ.h(node)
                gate_pairs.append([node, ngbr])
        self.gate_pairs = gate_pairs
        if self.state_sim:
            self.circuit = circ
            return
        circ.measure(q_reg, c_reg)
        self.circuit = circ

    def _build_Scontrolled_phase_gate(self, graph):
        unitary_op = Operator(np.identity(4))
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        ops_labels = []
        for node, ngbrs in graph.adjacency():
            for ngbr, _ in ngbrs.items():
                circ.h(node)
                circ.cx(node, ngbr)
                if bool(np.random.choice(2)):
                    label = 'unitary_{}_{}'.format(node, ngbrs)
                    ops_labels.append(label)
                    circ.unitary(unitary_op, [node, ngbr], label=label)
                circ.h(node)
                ops_labels.append([node, ngbr])
        self.ops_labels = ops_labels
        if self.state_sim:
            self.circuit = circ
            return
        circ.measure(q_reg, c_reg)
        self.circuit = circ

    def get_sorted_db(self):
        sorted_dict = dict([(i, []) for i in range(1,len(self.graph_db.keys()))])
        for _, itm in self.graph_db.items():
            if itm["G"] is not None:
                sorted_key = len(itm["G"].nodes)
                sorted_dict[sorted_key].append(itm["G"])
        return sorted_dict


if __name__ == "__main__":
    # Build example set of random circuits
    circ_builder = GraphCircuit()
    for _ in range(20):
        circ_builder.build_circuit()
