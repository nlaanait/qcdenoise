import numpy as np
import qiskit as qk
from datetime import datetime

import .graph_states import GraphDB

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
        for p in partition(n-i, i):
             yield (i,) + p


class CircuitConstructor:
    """Base class of constructing circuits
    
    Raises:
        NotImplementedError: build_circuit() must be overriden
        NotImplementedError: estimate_entanglement() must be overriden
    """
    def __init__(self,n_qubits= 2,
                n_shots=1024,
                verbose=True, save_to_file=False):
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.runtime = datetime.now().strftime("%Y_%m_%d")
        self.statevec = None
        self.circuit =  None
    
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
    def __init__(self, graph_db, gate_type="Controlled_Phase", smallest_subgraph=2, 
                       largest_subgraph=None, **kwargs):
        super(GraphCircuit, self).__init__(**kwargs)
        if isinstance(graph_db, GraphDB):
            self.graph_db = graph_db
        else
            self.graph_db = GraphDB()
        self.state_vec = True
        self.gate_type = gate_type
        self.all_graphs = None
        self.largest_subgraph = largest_subgraph
        self.smallest_subgraph = smallest_subgraph

    def generate_all_subgraphs(self):
        combs = list(set(partitions(target_qubits, start=smallest_subgraph)))
        sorted_dict = self.get_sorted_db()
        if largest_subgraph is None:
            for key, itm in sorted_dict.items():
                if len(itm) > 1:
                    largest_subgraph = key
        for (itm, comb) in enumerate(combs):
            if any([itm > largest_subgraph for itm in comb]):
                combs.pop(itm)
        sorted_dict = get_sorted_db(graph_db)
        graphs=[]
        for comb in combs:
            print('comb', comb)
            sub_graphs = []
            comb = list(comb)
            if len(comb) > 1:
                idx = 0
                while idx < len(comb):
                    all_graphs = sorted_dict[comb[idx]]
                    if len(all_graphs) < 1:
                        break
                    graph_idx = random.randint(0, len(all_graphs)-1)
                    sub_graphs.append(all_graphs[graph_idx])
                    idx += 1
    #             print('comb={}, subgraphs={}'.format(comb, sub_graphs))
            else:
                all_graphs = sorted_dict[comb[0]]
                if len(all_graphs) > 0:
                    graph_idx = random.randint(0, len(all_graphs)-1)
                    sub_graphs.append(all_graphs[graph_idx])
    #                 print('comb={}, subgraphs={}'.format(comb, sub_graphs))
            graphs.append(sub_graphs)
        self.all_graphs = graphs

    def combine_subgraphs(self, sub_graphs):
        union_graph = nx.DiGraph()
        for sub_g in sub_graphs:
            union_nodes = len(union_graph.nodes)
            if union_nodes > 1:
                first_node = random.randint(0, union_nodes - 1)
                offset = len(sub_g.nodes)
                second_node = random.randint(union_nodes, offset + union_nodes - 1)    
            union_graph = nx.disjoint_union(union_graph, sub_g)
            if union_nodes > 1:
                union_graph.add_weighted_edges_from([(first_node, second_node, 1.0)])
        return union_graph

    def build_circuit(self):
        if self.gate_type == "Controlled_Phase"
            self._build_controlled_phase_gate()


    def _build_controlled_phase_gate(self, graph):
        q_reg = qiskit.QuantumRegister(self.n_qubits)
        c_reg = qiskit.ClassicalRegister(self.n_qubits)
        self.circuit = qiskit.QuantumCircuit(q_reg, c_reg)
        for node, ngbrs in graph.adjacency():
            for ngbr, edge_attr in ngbrs.items():
                self.circuit.h(node)
                self.circuit.cx(node, ngbr)
                self.circuit.h(node)
                self.print_verbose("Adding Controlled Phase Gate between Node:{} and Neighbor:{}"\
                    .format(gate_type, node, ngbr)) 
        if self.statevec is None:
            return 
        self.circuit.measure(q_reg, c_reg)

    def combine_subgraphs(self, smallest_subgraph=2, largest_subgraph=None):
        combs = list(set(partitions(self.n_qubits, start=smallest_subgraph)))
        sorted_dict = get_sorted_db(graph_db)
        if largest_subgraph is None:
            for key, itm in sorted_dict.items():
                if len(itm) > 1:
                    largest_subgraph = key
        for (itm, comb) in enumerate(combs):
            if any([itm > largest_subgraph for itm in comb]):
                combs.pop(itm)
        comb = list(combs[random.randint(0, len(combs) - 1)])
        sub_graphs = []
        if len(comb) > 1:
            idx = 0
            while idx < len(comb):
                all_subgraphs = sorted_dict[comb[idx]]
                graph_idx = random.randint(0, len(all_subgraphs) - 1)
                sub_graphs.append(all_subgraphs[graph_idx])
                idx += 1
        else:
            all_subgraphs = sorted_dict[comb[0]]
            graph_idx = random.randint(0, len(all_graphs) - 1)
            sub_graphs.append(all_graphs[graph_idx])
        

        union_graph = nx.DiGraph()
        for sub_g in sub_graphs:
            union_nodes = len(union_graph.nodes)
            if union_nodes > 1:
                first_node = random.randint(0, union_nodes - 1)
                offset = len(sub_g.nodes)
                second_node = random.randint(union_nodes, offset + union_nodes - 1)    
            union_graph = nx.disjoint_union(union_graph, sub_g)
            if union_nodes > 1:
                union_graph.add_weighted_edges_from([(first_node, second_node, 1.0)])
        return union_graph, sub_graphs

    def get_sorted_db(self):
        sorted_dict = dict([(i, []) for i in range(1,len(self.graph_db.keys()))])
        for key, itm in self.graph_db.items():
            if itm["G"] is not None:
                sorted_key = len(itm["G"].nodes)
                sorted_dict[sorted_key].append(itm["G"])
        return sorted_dict
     
    
if __name__ == "__main__":
    # Build example set of random circuits
    circuit_name = 'graph'
    n_qubits = 5
    circ_constructor = CircuitConstructor(circuit_name=circuit_name,\
                    n_qubits=n_qubits,verbose=False,save_to_file=True)
    for i in range(1):
        print('sample %d' %i)
        circ_constructor.build_random_circuit()
        circ_constructor.save_circuit()
        # res = circ_sampler.execute_circuit()
        # measure entropy of circuit
