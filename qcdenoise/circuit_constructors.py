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
    def __init__(self,n_qubits=2, n_shots=1024, verbose=False, 
                 state_simulation=True, save_to_file=False):
        assert n_qubits >= 2, "# of qubits must be 2 or larger"
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.runtime = datetime.now().strftime("%Y_%m_%d")
        self.statevec = None
        self.circuit =  None
        self.state_sim = state_simulation
    
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
        self.statevec=qk.execute(self.circuit,\
                                backend=qk.Aer.get_backend('statevector_simulator'),
                                shots=self.n_shots).result().get_statevector()
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
    def __init__(self, graph_db=GraphDB(), gate_type="Controlled_Phase", stochastic=True, smallest_subgraph=2, 
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
        self.stochastic = stochastic
        self.circuit_name = "GraphState"

    def check_largest(self, val):
        for key, itm in self.all_graphs.items():
                if len(itm) != 0:
                    max_subgraph = key 
        if val is None:
            return max_subgraph 
        elif val > max_subgraph:
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
                first_node = random.randint(0, union_nodes - 1)
                offset = len(sub_g.nodes)
                second_node = random.randint(union_nodes, offset + union_nodes - 1)    
                union_graph = nx.disjoint_union(union_graph, sub_g)
                union_graph.add_weighted_edges_from([(first_node, second_node, 1.0)]) 
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
        if graph_plot and _plots:
            nx.draw_circular(circuit_graph, **nx_plot_options)
        # 3. Build a circuit from the graph state
        if self.gate_type == "Controlled_Phase":
            if self.stochastic:
                self.print_verbose("Assigning a Stochastic Controlled Phase Gate (H-CNOT-P(U)-H) to Node Edges")
            else:
                self.print_verbose("Assigning a Controlled Phase Gate (H-CNOT-H) to Node Edges")
            self._build_controlled_phase_gate(circuit_graph)

    def _build_controlled_phase_gate(self, graph):
        unitary_op = Operator(np.identity(4))
        ops_labels = []
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        for node, ngbrs in graph.adjacency():
            for ngbr, _ in ngbrs.items():
                circ.h(node)
                circ.cx(node, ngbr)
                if bool(np.random.choice(2)):
                    label = 'unitary_{}_{}'.format(node, ngbr)
                    ops_labels.append(label)
                    circ.unitary(unitary_op, [node, ngbr], label=label)
                circ.h(node)
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
     

class GHZCircuit(CircuitConstructor):
    def __init__(self, stochastic=True, **kwargs):
        super(GHZCircuit, self).__init__(**kwargs)
        self.ops_labels = None
        self.stochastic = stochastic
        self.circuit_name = "GHZ"

    def build_circuit(self):
        unitary_op = Operator(np.identity(4))
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        ops_labels = []
        circ.h(0) #pylint: disable=no-member
        self.ops_labels = []
        for q in range(self.n_qubits-1):
            circ.cx(q, q+1) #pylint: disable=no-member
            if self.stochastic and bool(np.random.choice(2)):
                label = 'unitary_{}_{}'.format(q,q+1)
                ops_labels.append(label)
                circ.unitary(unitary_op, [q, q+1], label=label) #pylint: disable=no-member
        self.ops_labels = ops_labels
        if self.state_sim:
            self.circuit = circ
            return
        circ.measure(q_reg, c_reg) #pylint: disable=no-member
        self.circuit = circ


class UCCSDCircuit(CircuitConstructor):
    def __init__(self, stochastic=True, **kwargs):
        super(UCCSDCircuit, self).__init__(**kwargs)
        self.ops_labels = None
        self.stochastic = stochastic
        self.circuit_name = "UCCSD"
    
    def build_circuit(self):
        unitary_op = Operator(np.identity(4))
        theta = np.pi * np.random.rand(self.n_qubits)
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        for idx, q in enumerate(q_reg):
            circ.x(q) #pylint: disable=no-member
            circ.ry(theta[idx],q) #pylint: disable=no-member
        ops_labels = []
        for (idx, q), q_next in zip(enumerate(q_reg), q_reg[1:]):
            circ.cry(theta[idx], q, q_next) #pylint: disable=no-member
            circ.cx(q, q_next) #pylint: disable=no-member
            # for now not considering noisy cnot gates in controlled-Y
            if self.stochastic and bool(np.random.choice(2)):
                label = 'unitary_{}_{}'.format(idx, idx+1)
                ops_labels.append(label)
                circ.unitary(unitary_op, [idx, idx+1], label=label) #pylint: disable=no-member
        self.ops_labels = ops_labels
        if self.state_sim:
            self.circuit = circ
            return
        circ.measure(q_reg, c_reg) #pylint: disable=no-member
        self.circuit = circ


if __name__ == "__main__":
    # Build example set of random graph circuits
    circ_builder = GraphCircuit()
    for _ in range(20):
        circ_builder.build_circuit()
    # Build example set of stochastic GHZ circuits
    circ_builder = GHZCircuit()
    for _ in range(20):
        circ_builder.build_circuit()
