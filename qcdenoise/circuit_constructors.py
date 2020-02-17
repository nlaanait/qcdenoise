class CircuitConstructor:
    def __init__(self):
        pass

    def initialize_subgraphs(self):
        """ Builds a set of interesting subgraphs.
        """
        pass 

    def _build_from_gate_set(self,gate_set):
        '''
        first draft of building a circuit constructor from a list of gates
        pass gate_set as a list of dictionaries (yes/no?  other?)
        
        [{'gate':'I' or 'H' or 'CNOT','qubits': single or tuple}]
        ex: ['I','I','I','I'] = [{'gate':'I','qubit':0},{'gate':'I','qubit':1},\
                                    {'gate':'I','qubit':2},{'gate':'I','qubit':3}]
        ex (GHZ-3): ['H','CNOT','CNOT'] = [{'gate':'H','qubit':0},{'gate':'CNOT','qubit':(0,1)},\
                                            {'gate':'CNOT','qubit':(1,2)}]
        ex (C-3): ['H','CNOT','H','CNOT','H','H'] = [{'gate':'H','qubit':0},{'gate':'CNOT','qubit':(0,1)},\
                                            {'gate':'H','qubit':1},{'gate':'CNOT',\
                                            'qubit':(1,2)},{'gate':'H','qubit':0},{'gate':'H','qubit':2}]
        '''
    
        q_reg = QuantumRegister(self.n_qubits)
        c_reg = ClassicalRegister(self.n_qubits)
        circ = QuantumCircuit(q_reg, c_reg)
        if self.insert_unitary:
            # excite one of the qubits to |1> 
            idx = np.random.randint(0, self.n_qubits)
            circ.initialize([0,1], q_reg[idx]) #pylint: disable=no-member
        self.ops_labels = []
        for gdx in range(len(gate_set)):
            if gate_set[gdx]['gate'] is 'H':
                qbx=gate_set[gdx]['qubit']
                circ.h(qr[qbx])
            elif gate_set[gdx]['gate'] is 'CNOT':
                src,tgt=gate_set[gdx]['qubit']
                circ.cx(qr[qr[src],qr[tgt])
                if self.insert_unitary and bool(np.random.choice(2)):
                    label = 'unitary_{}_{}'.format(q,q+1)
                    self.ops_labels.append(label)
                    circ.unitary(self.unitary_op, [q, q+1], label=label)
            elif gate_set[gdx]['gate'] is 'I':
                print('identity gate is trivial')
        circ.measure(q_reg, c_reg) #pylint: disable=no-member
        return circ
    
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