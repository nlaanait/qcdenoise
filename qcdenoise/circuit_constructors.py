import random
import warnings
from datetime import datetime
from time import sleep

import networkx as nx
from networkx.algorithms.approximation import vertex_cover
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

def sigma_prod(op_str):
    from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product
    from sympy import I
    pauli_dict= {"I":"I","X":Pauli(1), "Y":Pauli(2),"Z":Pauli(3)}
    pauli_label = {Pauli(1):"X",Pauli(2):"Y",Pauli(3):"Z","I":"I"}
    op_list=list(op_str)
    if ('X' not in op_str) and ('Y' not in op_str) and ('Z' not in op_str):
        mat3='I'
        coef=np.complex(1,0)
    pauli_list = [pauli_dict[x] for x in op_list]
    coef_list = []
    while len(pauli_list)>1:
        mat1=pauli_list.pop()
        mat2=pauli_list.pop()
        if mat1==mat2:
            mat3='I'
            coef=np.complex(1,0)
        elif 'I' not in [mat1,mat2]:
            mat3=evaluate_pauli_product(mat2*mat1).args[-1]
            coef=evaluate_pauli_product(mat2*mat1).args[:-1]
            if coef==(I,):
                coef=np.complex(0,1)
            elif coef==(-1,I):
                coef=np.complex(0,-1)
            else:
                coef=np.complex(1,0)
        else:
            mat3=[x for x in [mat1,mat2] if x!='I'][0]
            coef=np.complex(1,0)
        coef_list.append(coef)
        pauli_list.append(mat3)
    return np.prod(np.asarray(coef_list)),[pauli_label[x] for x in pauli_list][0]
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

    def get_statevector(self):
        """execute circuit with state_vector sim."""
        self.statevec=qk.execute(self.circuit,\
                                backend=qk.Aer.get_backend('statevector_simulator'),
                                shots=self.n_shots).result().get_statevector()
        return

    def estimate_entanglement(self):
        """estimate entanglement of final state using n-qubit entanglement
        winessess if circuit was prepared as GHZ state then assume maximally
        entangled if circuit was prepared as random graph state then use
        witnesses."""
        raise NotImplementedError


class GraphCircuit(CircuitConstructor):
    """Class to construct circuits from graph states.

    Arguments:
        CircuitConstructor {Parent class} -- abstract class
    """
    def __init__(self, graph_data=None, gate_type="Controlled_Phase", stochastic=True, smallest_subgraph=2,
                       largest_subgraph=None, **kwargs):
        super(GraphCircuit, self).__init__(**kwargs)
        if graph_data is None:
            self.graph_db = GraphDB()
        else:
            self.graph_db = GraphDB(graph_data=graph_data)
        self.gate_type = gate_type
        self.all_graphs = self.get_sorted_db()
        self.largest_subgraph = self.check_largest(largest_subgraph)
        self.smallest_subgraph = max(2, smallest_subgraph)
        self.graph_combs = self.generate_all_subgraphs()
        self.ops_labels = None
        self.stochastic = stochastic
        self.name = "GraphState"
        self.circuit_graph = None

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
        #union_graph = nx.DiGraph()
        union_graph = nx.Graph()
        for sub_g in sub_graphs:
            union_nodes = len(union_graph.nodes)
            if union_nodes > 1:
                first_node = random.randint(0, union_graph.order() - 1)
                second_nodes = np.random.randint(union_graph.order(), sub_g.order() + union_graph.order() - 1,
                                                 sub_g.order())
                union_graph = nx.disjoint_union(union_graph, sub_g)
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
            temp[idx]='X'
            for jdx in self.circuit_graph.neighbors(idx):
                temp[jdx]='Z'
            temp = "".join(temp)
            generators.append(temp)
        self.generators = generators

    def get_stabilizers(self):
        """ get the stabilizer operators for a graph state """
        binary_keys = [np.binary_repr(x,self.n_qubits) for x in range(2**self.n_qubits)]
        stab_label = []
        for idx in binary_keys:
            coefs = [int(x) for x in list(idx)]
            op_mat = []
            for jdx in range(len(coefs)):
                if coefs[jdx]==0:
                    op_mat.append(list('I'*self.n_qubits))
                else:
                    op_mat.append(list(self.generators[jdx]))
            op_mat = np.asarray(op_mat)
            cf_arr = []
            lb_arr = []
            for kdx in range(op_mat.shape[0]):
                cf,lb = sigma_prod(''.join(op_mat[:,kdx]))
                cf_arr.append(cf)
                lb_arr.append(lb)
            if np.iscomplex(np.prod(cf_arr)):
                print("Flag-error, coefficient cannot be complex")
                return
            else:
                val = np.prod(cf_arr)
                if np.real(val)==1:
                    stab_label.append('+'+''.join(lb_arr))
                else:
                    stab_label.append('-'+''.join(lb_arr))
        self.stabilizers = stab_label

    def _build_controlled_phase_gate(self, graph):
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        gate_pairs = []
        for node, ngbrs in graph.adjacency():
            for ngbr, _ in ngbrs.items():
                circ.h(node)
                circ.cx(node, ngbr)
                if bool(np.random.choice(2)) and self.stochastic:
                    label = 'unitary_{}_{}'.format(node, ngbr)
                    ops_labels.append(label)
                    circ.unitary(unitary_op, [node, ngbr], label=label)
                circ.h(node)
        if self.stochastic: self.ops_labels = ops_labels
        if self.state_sim:
            self.circuit = circ
            return
        circ.barrier()
        circ.measure(q_reg, c_reg)
        self.circuit = circ

    def get_sorted_db(self):
        sorted_dict = dict([(i, []) for i in range(1, len(self.graph_db.keys()) \
            + 1)])
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
        self.name = "GHZ"

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
        circ.barrier()
        circ.measure(q_reg, c_reg) #pylint: disable=no-member
        self.circuit = circ


class UCCSDCircuit(CircuitConstructor):
    def __init__(self, stochastic=True, **kwargs):
        super(UCCSDCircuit, self).__init__(**kwargs)
        self.ops_labels = None
        self.stochastic = stochastic
        self.name = "UCCSD"

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
        circ.barrier()
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
