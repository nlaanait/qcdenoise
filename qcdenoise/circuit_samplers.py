from warnings import warn
import io
import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.quantum_info.operators import Operator

global seed
seed = 1234
np.random.seed(seed)

class CircuitSampler:
    def __init__(self, n_qubits= 2, circuit_name='GHZ', n_shots=1024, noise_specs={'class': 'unitary_noise', 'type':'phase_amplitude_damping_error', 
                                                  'max_prob':0.1, 'unitary_op':None}, verbose=True):
        self.circuit_name = circuit_name
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.verbose = verbose
        self.callbacks = {'phase_amplitude_damping_error': CircuitSampler.get_phase_amp_damp_error,
                    'amplitude_damping_error': CircuitSampler.get_amp_damp_error}
        self.insert_unitary = False
        self.noise = False
        self.noise_specs = noise_specs if noise_specs is not None else None
        self.noise_model = NoiseModel()
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
                        getattr(errors, self.error_type)
                        self.unitary_op = self.noise_specs.pop('unitary_op')
                        if self.unitary_op is None:
                            self.unitary_op = Operator(np.identity(4))
                        else:
                            assert isinstance(self.unitary_op, Operator), "unitary op must be instance of qiskit.quantum_info.operators.Operator"
                        self.print_verbose("Using Unitary Noise Model ({})".format(self.error_type))
                    except AttributeError as e:
                        print(e)
                else:
                    raise NotImplementedError("Noise model: '{}' is not implemented".format(self.noise_class))
            else:
                self.print_verbose("Ideal Circuit Simulation")
    
    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def build_circuit(self):
        if self.circuit_name == 'GHZ':
            self.circuit = self._build_GHZ()
        elif self.circuit_name == 'UCCSD':
            self.circuit = self._build_UCCSD()
        else:
            raise NotImplementedError("circuit: {} has not been implemented.".format(self.circuit_name))

    def build_noise_model(self):
        if self.noise:
            if self.noise_class == 'unitary_noise':
                self.noise_model = self._build_unitary_noise_model()

    def execute_circuit(self):
        job = execute(self.circuit, QasmSimulator(), noise_model= self.noise_model, seed_simulator=seed, shots=self.n_shots)
        result = job.result()
        prob_dict = result.get_counts()
        binary_strings = CircuitSampler.generate_binary_strings(self.n_qubits)
        full_prob_dict = dict([(key, 0) for key in binary_strings])
        for key, itm in prob_dict.items():
            full_prob_dict[key] = itm / self.n_shots
        return full_prob_dict
        
    def get_prob_vector(self, ideal=True):
        """Returns the output probability vector of the sampled circuit.
        
        Keyword Arguments:
            ideal {bool} -- if True the circuit is executed twice: 1. specified noise and 2. w/o noise (default: {True})

        Returns:
            numpy.ndarray -- probability vector with shape (2**n_qubits,1) or (2**n_qubits,2) if 'ideal'
        """
        noise_prob = self.execute_circuit()
        if ideal:
            old_max_prob = self.noise_specs['max_prob']
            self.noise_specs['max_prob'] /= 100
            self.build_noise_model()
            ideal_prob = self.execute_circuit()
            self.noise_specs['max_prob'] = old_max_prob
            prob_arr = np.array([[noise_val, ideal_val] for (_, noise_val), (_, ideal_val) in zip(noise_prob.items(), ideal_prob.items())])
            return prob_arr
        prob_arr = np.array([noise_val for (_, noise_val) in noise_prob.items()]) 
        return prob_arr
        
    def _build_from_graph(self):
        pass
    
    def _build_GHZ(self):
        q_reg = QuantumRegister(self.n_qubits)
        c_reg = ClassicalRegister(self.n_qubits)
        circ = QuantumCircuit(q_reg, c_reg)
        if self.insert_unitary:
            # excite one of the qubits to |1> 
            idx = np.random.randint(0, self.n_qubits)
            circ.initialize([0,1], q_reg[idx]) #pylint: disable=no-member
        circ.h(0) #pylint: disable=no-member
        self.ops_labels = []
        for q in range(self.n_qubits-1):
            circ.cx(q, q+1) #pylint: disable=no-member
            if self.insert_unitary and bool(np.random.choice(2)):
                label = 'unitary_{}_{}'.format(q,q+1)
                self.ops_labels.append(label)
                circ.unitary(self.unitary_op, [q, q+1], label=label) #pylint: disable=no-member
        ## TODO: must check if built circuit is unitary
        circ.measure(q_reg, c_reg) #pylint: disable=no-member
        return circ

    def _build_UCCSD(self):
        theta = np.pi * np.random.rand(self.n_qubits)
        q_reg = QuantumRegister(self.n_qubits)
        c_reg = ClassicalRegister(self.n_qubits)
        circ = QuantumCircuit(q_reg, c_reg)
        for idx, q in enumerate(q_reg):
            circ.x(q) #pylint: disable=no-member
            circ.ry(theta[idx],q) #pylint: disable=no-member
        self.ops_labels = []
        for (idx, q), q_next in zip(enumerate(q_reg), q_reg[1:]):
            circ.cry(theta[idx], q, q_next) #pylint: disable=no-member
            circ.cx(q, q_next) #pylint: disable=no-member
            # for now not considering noisy cnot gates in controlled-Y
            if self.insert_unitary and bool(np.random.choice(2)):
                label = 'unitary_{}_{}'.format(idx, idx+1)
                self.ops_labels.append(label)
                circ.unitary(self.unitary_op, [idx, idx+1], label=label) #pylint: disable=no-member
        ## TODO: must check if built circuit is unitary
        circ.measure(q_reg, c_reg) #pylint: disable=no-member
        return circ 

    def _build_unitary_noise_model(self):
        error_call = getattr(errors, self.error_type)
        noise_model = NoiseModel()
        if self.ops_labels:
            for op in self.ops_labels:
                error, params = self.callbacks[self.error_type](error_call, **self.noise_specs)
                self.print_verbose(params) 
                noise_model.add_all_qubit_quantum_error(error, op)
            noise_model.add_basis_gates(['unitary'])
            return noise_model
        return noise_model

    def get_adjacency_tensor(self, max_tensor_dims=16, basis_gates=['id', 'cx', 'u1', 'u2', 'u3'], fixed_size=True):
        """[summary]
        
        Keyword Arguments:
            num_dims {int} -- [description] (default: {16})
            basis_gates {[type]} -- [description] (default: {None})
        
        Returns:
            [type] -- [description]
        """
        def get_dag(**kwargs):
            """Transpiler callable
            
            Returns:
                qiskit.dagcircuit.dagcircuit -- Directed Acyclic Graph representation of transpiled circuit
            """
            global dag
            dag = kwargs['dag']
            return dag
        if self.noise_model:
            basis_gates = self.noise_model.basis_gates
            self.print_verbose('Using noise model basis gates in transpilation.')
        # decompose() is needed to force transpiler to go into custom unitary gate ops
        _ = transpile(self.circuit.decompose(), basis_gates=basis_gates, optimization_level=0, callback=get_dag)
        adj_T = CircuitSampler.generate_adjacency_tensor(dag, adj_tensor_dim=(max_tensor_dims, self.n_qubits, self.n_qubits), encoding=None, fixed_size=fixed_size)
        return adj_T

    @staticmethod
    def get_phase_amp_damp_error(func, max_prob=0.1):
        phases = np.random.uniform(high=max_prob, size=2)
        amps = [min(0, 0.5-phase) for phase in phases] # hack to force CPTP
        q_error_1 = func(phases[0], amps[0])
        q_error_2 = func(phases[1], amps[1])
        unitary_error = q_error_1.tensor(q_error_2) 
        return unitary_error, (phases, amps)

    @staticmethod
    def get_amp_damp_error(func, max_prob=0.1):
        amps = np.random.uniform(high=max_prob, size=2)
        q_error_1 = func(amps[0])
        q_error_2 = func(amps[1])
        unitary_error = q_error_1.tensor(q_error_2) 
        return unitary_error, amps

    @staticmethod
    def generate_binary_strings(n_qubits):
        global string
        string = io.StringIO()
        # recursive func
        def generate_bitstring(n_qubits, bit_string, i):  
            if i == n_qubits:
                global string
                s = ''
                print(s.join(bit_string), file=string)
                return
            bit_string[i] = '0'
            generate_bitstring(n_qubits, bit_string, i + 1)
            bit_string[i] = '1'
            generate_bitstring(n_qubits, bit_string, i + 1)
        bit_string = [None]*n_qubits
        # recursive call and parse string buffer
        generate_bitstring(n_qubits, bit_string, 0)
        s = string.getvalue()
        binary_strings = s.split('\n')
        binary_strings.pop()
        return binary_strings

    @staticmethod
    def generate_adjacency_tensor(dag, adj_tensor_dim, encoding=None, fixed_size=True):
        """[summary]
       
        Arguments:
            dag {[type]} -- [description]
            adj_tensor_dim {[type]} -- [description]
        
        Keyword Arguments:
            encoding {[type]} -- [description] (default: {None})
            trim {bool} -- [description] (default: {True})
        
        Returns:
            [type] -- [description]
        """
        adj_tensor = np.zeros(adj_tensor_dim)
        encoding = encoding if encoding else {'id': 0, 'cx': 1, 'u1': 2, 'u2':3, 'u3': 4}
        assert isinstance(dag, qiskit.dagcircuit.DAGCircuit), 'dag must be an instance of qiskit.dagcircuit.dagcircuit'
        for gate in dag.gate_nodes():
            qubits = gate.qargs
            if qubits:
                if len(qubits) == 1:
                    q_idx = qubits[0].index
                    plane_idx = 0
                    write_success = False
                    while plane_idx < adj_tensor.shape[0]:
                        if adj_tensor[plane_idx, q_idx, q_idx] == 0:
                            adj_tensor[plane_idx, q_idx, q_idx] = encoding[gate.name]
                            write_success = True
                            break
                        else:
                            plane_idx +=1
                    if not write_success:
                        warn('max # of planes in the adjacency tensor have been exceeded. Initialize a larger adjacency tensor')
                if len(qubits) == 2:
                    q_idx_1, q_idx_2 = [q.index for q in qubits]
                    plane_idx = 0
                    write_success = False
                    while plane_idx < adj_tensor.shape[0]:
                        if adj_tensor[plane_idx, q_idx_1, q_idx_2] == 0: # assume that the graph is undirected
                            adj_tensor[plane_idx, q_idx_1, q_idx_2] = encoding[gate.name]
                            adj_tensor[plane_idx, q_idx_2, q_idx_1] = encoding[gate.name]
                            write_success = True
                            break
                        else:
                            plane_idx += 1
                    if not write_success:
                        warn('max # of planes in the adjacency tensor have been exceeded. Initialize a larger adjacency tensor')

        if not fixed_size:
            # get rid of planes in adj_tensor with all id gates (i.e zeros)
            all_zeros = np.zeros_like(adj_tensor[0])
            adj_tensor = np.array([adj_plane for adj_plane in adj_tensor if np.any(adj_plane != all_zeros)])
        return adj_tensor


if __name__ == "__main__":
    # Build 10-qubit GHZ with randomly inserted unitary operators and noised'em up
    noise_specs = {'class':'unitary_noise', 
                    'type':'phase_amplitude_damping_error', 
                    # 'type':'amplitude_damping_error', 
                    'max_prob': 0.35, 
                    'unitary_op':None}
    circuit_name = 'GHZ'
    n_qubits = 5
    circ_sampler = CircuitSampler(circuit_name=circuit_name, n_qubits=n_qubits, noise_specs=noise_specs, verbose=False)
    for i in range(10):
        print('sample %d' %i)
        circ_sampler.build_circuit()
        circ_sampler.build_noise_model()
        # res = circ_sampler.execute_circuit()
        prob = circ_sampler.get_prob_vector()
        print(prob)
     

