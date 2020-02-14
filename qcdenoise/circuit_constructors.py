class CircuitConstructor:
    def __init__(self):
        pass

    def initialize_subgraphs(self):
        """ Builds a set of interesting subgraphs.
        """
        pass 

    def build_graph(self):
        """Build a graph from a set of sub-graphs
        """
        pass

    def build_circuit(self):
        """build qiskit circuit from graph
            Reuse this for CircuitSampler()
        """
        pass

    def execute_circuit(self):
        """execute circuit with state_vector sim 
        """ 
        pass

    def calculate_entropy(self):
        """get entropy of final state
        """
        pass