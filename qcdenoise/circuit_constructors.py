import numpy as np
import qiskit as qk
import qiskit.quantum_info.operators as qops
import circuit_samplers as cs

class CircuitConstructor:
    def __init__(self,n_qubits= 2, circuit_name='GHZ', n_shots=1024,\
                 noise_specs={'class': 'unitary_noise',\
                'type':'phase_amplitude_damping_error',\
                        'max_prob':0.1, 'unitary_op':None}, verbose=True):
        self.circuit_name = circuit_name
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.verbose = verbose
        # remove noise and error unitary stuff
        
        self.callbacks = {'phase_amplitude_damping_error': cs.CircuitSampler.get_phase_amp_damp_error,
                    'amplitude_damping_error': cs.CircuitSampler.get_amp_damp_error}
        self.insert_unitary = False
        self.noise = False
        self.noise_specs = noise_specs if noise_specs is not None else None
        self.noise_model = qk.aer.noise.NoiseModel()
        # check attributes 
        assert n_qubits >= 2, "# of qubits must be 2 or larger"
        # populate noise model attributes and do checks on model specs
        if self.noise_specs:
            self.noise = True
            self.error_type = self.noise_specs.pop('type')
            if self.noise_specs:
                self.noise_class = self.noise_specs.pop('class')
                if self.noise_class == 'unitary_noise':
                    self.insert_unitary = True
                    assert self.error_type in list(self.callbacks.keys()), "noise 'type' must be one of: {}".format(list(self.callbacks.keys()))
                    try:
                        getattr(qk.aer.noise.errors, self.error_type)
                        self.unitary_op = self.noise_specs.pop('unitary_op')
                        if self.unitary_op is None:
                            self.unitary_op = qops.Operator(np.identity(4))
                        else:
                            assert isinstance(self.unitary_op, qops.Operator), "unitary op must be instance of qiskit.quantum_info.operators.Operator"
                        self.print_verbose("Using Unitary Noise Model ({})".format(self.error_type))
                    except AttributeError as e:
                        print(e)
                else:
                    raise NotImplementedError("Noise model: '{}' is not implemented".format(self.noise_class))
            else:
                self.print_verbose("Ideal Circuit Simulation")

    def _add_Hgate(self,qubit):
        self.gate_list.append({'gate':'H','qubit':qubit})
        return
    
    def _add_CNOT(self,src,tgt):
        self.gate_list.append({'gate':'CNOT','qubits':(src,tgt)})
        
    def initialize_subgraphs(self):
        """ Builds a set of interesting subgraphs.
        """
        pass 
    def onequbit_circuit_sample():
        """ randomly sample from the set of all one qubit subcircuits
        that we are interested in 
        """
        onequbit_circuits={0:[{'gate':'I','qubits':0}],\
                            1:[{'gate':'H','qubits':0}]}
        return onequbit_circuits[np.random.choice(onequbit_circuits.keys())]
        
    def twoqubit_circuit_sample():
        """ randomly sample from the set of all two qubit subcircuits
        that we are interested in
        
        returns list of dictionaries that are gate set lists
        """
        twoqubit_circuits={\
            0:[{'gate':'I','qubits':0},{'gate':'I','qubits':1}],\
            1:[{'gate':'I','qubits':0},{'gate':'H','qubits':1}],\
            2:[{'gate':'H','qubits':0},{'gate':'I','qubits':1}],\
            3:[{'gate':'H','qubits':0},{'gate':'H','qubits':1}],\
            4:[{'gate':'H','qubits':0},{'gate':'CNOT','qubits':(0,1)},{'gate':'H','qubits':1}],\
            5:[{'gate':'H','qubits':1},{'gate':'CNOT','qubits':(1,0)},{'gate':'H','qubits':0}]
            }
        return twoqubit_circuits[np.random.choice(list(twoqubit_circuits.keys()))]
    
    def threequbit_circuit_sample():
        threequbit_circuits={\
        0:[{'gate':'I','qubits':0},{'gate':'I','qubits':1},{'gate':'I','qubits':2}],\
        1:[{'gate':'I','qubits':0},{'gate':'I','qubits':1},{'gate':'H','qubits':2}],\
        2:[{'gate':'I','qubits':0},{'gate':'H','qubits':1},{'gate':'I','qubits':2}],\
        3:[{'gate':'H','qubits':0},{'gate':'I','qubits':1},{'gate':'I','qubits':2}],\
        4:[{'gate':'H','qubits':0},{'gate':'I','qubits':1},{'gate':'H','qubits':2}],\
        5:[{'gate':'H','qubits':0},{'gate':'H','qubits':1},{'gate':'I','qubits':2}],\
        6:[{'gate':'H','qubits':0},{'gate':'I','qubits':1},{'gate':'H','qubits':2}],\
        7:[{'gate':'H','qubits':0},{'gate':'H','qubits':1},{'gate':'H','qubits':2}],\
        8:[{'gate':'H','qubits':0},{'gate':'CNOT','qubits':(0,1)},\
                    {'gate':'H','qubits':1},{'gate':'CNOT','qubits':(0,2)},\
                    {'gate':'H','qubits':2},{'gate':'H','qubits':0}],\
        9:[{'gate':'H','qubits':0},{'gate':'CNOT','qubits':(0,1)},{'gate':'CNOT','qubits':(1,2)}]
        }
        return threequbit_circuits[np.random.choice(list(threequbit_circuits.keys()))]

    def _graph_state(self):
        '''
        return gate set needed to construct a graph state on n-qubits

        Returns
        -------
        list of dictionaries.

        '''
        # check attributes 
        assert self.n_qubits > 2, "# of qubits must be 3 or larger (use Bell state for 2 qubits)"
        graph_state_gates=[]
        n_idx=0
        graph_state_gates.append({'gate':'H','qubits':n_idx})
        graph_state_gates.append({'gate':'CNOT','qubits':(n_idx,n_idx+1)})
        graph_state_gates.append({'gate':'H','qubits':n_idx+1})
        n_idx+=1
        while n_idx<(self.n_qubits-1):
            graph_state_gates.append({'gate':'CNOT','qubits':(n_idx,n_idx+1)})
            graph_state_gates.append({'gate':'H','qubits':n_idx+1})
            n_idx+=1
        graph_state_gates.append({'gate':'CNOT','qubits':(n_idx,0)})
        graph_state_gates.append({'gate':'H','qubits':0})
        return graph_state_gates
            
    def _build_from_gate_set(self,gate_set):
        '''
        This was adapted from the _build_GHZ() routine in circuit_sampler
        So I assumed that the object would have similar class attributes
        self.n_qubits, self.insert_unitary ... etc
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
        returns QIskit CircuitObj
        '''
        
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        if self.insert_unitary:
            # excite one of the qubits to |1> 
            idx = np.random.randint(0, self.n_qubits)
            circ.initialize([0,1], q_reg[idx]) #pylint: disable=no-member
        self.ops_labels = []
        for gdx in range(len(gate_set)):
            if gate_set[gdx]['gate']=='H':
                qbx=gate_set[gdx]['qubit']
                circ.h(q_reg[qbx])
            elif gate_set[gdx]['gate']=='CNOT':
                src,tgt=gate_set[gdx]['qubit']
                circ.cx(q_reg[src],q_reg[tgt])
            elif gate_set[gdx]['gate']=='I':
                print('identity gate is trivial')
        circ.measure(q_reg, c_reg) #pylint: disable=no-member
        return circ
    
    def build_graph(self):
        """Build a graph from a set of sub-graphs
        """
        pass

    def build_random_circuit(self):
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