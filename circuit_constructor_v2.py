import numpy as np
import qiskit as qk
from datetime import datetime

class CircuitConstructor:
    def __init__(self,n_qubits= 2, block_tuple = (2,1), \
                circuit_name='graph', n_shots=1024,\
                verbose=True,save_to_file=False):
        self.circuit_name = circuit_name
        self.n_qubits = n_qubits
        self.blocks = block_tuple
        self.n_shots = n_shots
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.runtime = datetime.now().strftime("%Y_%m_%d")
        self.statevec = None
        
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

    def _graph_state_dict():
        _2qubit=[{'gate'}]
        return
    def _random_graph_state(self):
        '''
        return gate set needed to construct a graph state on n-qubits
        Configurations are hard-wired for 2,3,4,5,6,7 qubits
        from Table V in arXiv:060296 (Hein et al.)
        Returns
        -------
        list of dictionaries.

        '''
        graph_state_gates=[]
        state = np.random.choice(range(self.n_qubits))
        if self.n_qubits==2:
            # construct star graph state (rooted tree at qubit 0)
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
        elif self.n_qubits==3:
            # construct linear graph state (CNOT chain)
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
        else:
            # placeholder for now - replace with random rewiring of star  graph
            # remove single edge from [0,n-1] and replace with edge from [m,n-1]
            pass
        return graph_state_gates

    def build_graph_sequence(self,index=1):
        """Build the i-th graph state circuit
        in the sequential rewiring of a star graph into a linear chain
        """
        graph_state_gates=[]
        n_edges=self.n_qubits-1
        # construct star graph state (rooted tree at qubit 0)
        n_idx=1
        graph_state_gates.append({'gate':'H','qubits':0})
        graph_state_gates.append({'gate':'CNOT','qubits':(0,n_idx)})
        graph_state_gates.append({'gate':'H','qubits':n_idx})
        n_idx+=1
        # connect up to <index> qubits to root qubit
        while n_idx<(n_edges-index):
            graph_state_gates.append({'gate':'CNOT','qubits':(0,n_idx+1)})
            graph_state_gates.append({'gate':'H','qubits':n_idx+1})
            n_idx+=1
        while n_idx<=n_edges:
            graph_state_gates.append({'gate':'CNOT','qubits':(n_idx,n_idx+1)})
            graph_state_gates.append({'gate':'H','qubits':n_idx+1})
        graph_state_gates.append({'gate':'CNOT','qubits':(n_idx,0)})
        graph_state_gates.append({'gate':'H','qubits':0})
           
    def _build_from_gate_set(self,gate_set,statevec=True):
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
        for gdx in range(len(gate_set)):
            if gate_set[gdx]['gate']=='H':
                qbx=gate_set[gdx]['qubits']
                circ.h(q_reg[qbx])
            elif gate_set[gdx]['gate']=='CNOT':
                src,tgt=gate_set[gdx]['qubits']
                circ.cx(q_reg[src],q_reg[tgt])
            elif gate_set[gdx]['gate']=='I':
                print('identity gate is trivial')
        if statevec:
            return circ
        else:
            circ.measure(q_reg, c_reg) #pylint: disable=no-member
            return circ

    def build_random_circuit(self,statevec=True):
        """build qiskit circuit from graph
            Reuse this for CircuitSampler()
        """
        random_gate_set=self._graph_state()
        self.circuit=self._build_from_gate_set(random_gate_set,statevec)
        return 
    
    def save_circuit(self):
        if self.save_to_file:
            with open('circuit_'+self.runtime+'.txt','w') as fp:
                for line in self.circuit.qasm():
                    fp.write(line)
        return

    def execute_circuit(self):
        """execute circuit with state_vector sim 
        """ 
        self.statevec=qk.execute(circ_constructor.circuit,\
                                 backend=qk.Aer.get_backend('statevector_simulator'),
                                 shots=10000).result().get_statevector()
        return

    def estimate_entanglement(self):
        """estimate entanglement of final state
        using n-qubit entanglement winessess
        if circuit was prepared as GHZ state then assume maximally entangled
        if circuit was prepared as random graph state then use witnesses
        """
        if self.circuit_name=='GHZ':
            return 1.0
        else:
            return 0.0
        pass
    
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
