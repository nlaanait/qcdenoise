import io
from copy import deepcopy
from warnings import warn

import numpy as np
import qiskit as qk
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.quantum_info.operators import Operator

from .circuit_constructors import CircuitConstructor, GHZCircuit

global seed
seed = 1234
np.random.seed(seed)


class CircuitSampler:
    def __init__(self, n_qubits=2, stochastic=True, circuit_builder=GHZCircuit, n_shots=1024, verbose=True):
        assert n_qubits >= 2, "# of qubits must be 2 or larger"
        self.n_qubits = n_qubits
        assert isinstance(circuit_builder, CircuitConstructor),\
            "circuit_builder must be an instance of the CircuitConstructor class"
        self.circ_builder = circuit_builder
        self.circ_builder.verbose = verbose
        self.circ_builder.state_sim = False
        self.n_shots = n_shots
        self.verbose = verbose
        self.noise_model = NoiseModel()
        self.circuit = None
        
    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def sample(self):
        raise NotImplementedError

    def execute_circuit(self):
        job = qk.execute(self.circuit, backend=qk.Aer.get_backend('qasm_simulator'), 
                         noise_model= self.noise_model, seed_simulator=seed, 
                         shots=self.n_shots)
        result = job.result()
        prob_dict = result.get_counts()
        binary_strings = CircuitSampler.generate_binary_strings(self.n_qubits)
        full_prob_dict = dict([(key, 0) for key in binary_strings])
        for key, itm in prob_dict.items():
            full_prob_dict[key] = itm / self.n_shots
        return full_prob_dict
        
    def get_adjacency_tensor(self, max_tensor_dims=16, basis_gates=['id', 'cx', 'u1', 'u2', 'u3'], 
                             fixed_size=True, undirected=True):
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
        _ = qk.transpile(self.circuit.decompose(), basis_gates=basis_gates, 
                         optimization_level=0, callback=get_dag)
        adj_T = CircuitSampler.generate_adjacency_tensor(dag, adj_tensor_dim=(max_tensor_dims, self.n_qubits, 
                                                         self.n_qubits), encoding=None, fixed_size=fixed_size, 
                                                         undirected=undirected)
        return adj_T

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
    def generate_adjacency_tensor(dag, adj_tensor_dim, encoding=None, fixed_size=True, undirected=True):
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
        assert isinstance(dag, qk.dagcircuit.DAGCircuit), 'dag must be an instance of qiskit.dagcircuit.dagcircuit'
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
                        warn('max # of planes in the adjacency tensor have been exceeded. Initialize a larger \
                             adjacency tensor')
                if len(qubits) == 2:
                    q_idx_1, q_idx_2 = [q.index for q in qubits]
                    plane_idx = 0
                    write_success = False
                    while plane_idx < adj_tensor.shape[0]:
                        if adj_tensor[plane_idx, q_idx_1, q_idx_2] == 0:
                            adj_tensor[plane_idx, q_idx_1, q_idx_2] = encoding[gate.name]
                            if undirected:
                                adj_tensor[plane_idx, q_idx_2, q_idx_1] = encoding[gate.name]
                            write_success = True
                            break
                        else:
                            plane_idx += 1
                    if not write_success:
                        warn('max # of planes in the adjacency tensor have been exceeded. Initialize a larger adjacency\
                             tensor')

        if not fixed_size:
            # get rid of planes in adj_tensor with all id gates (i.e zeros)
            all_zeros = np.zeros_like(adj_tensor[0])
            adj_tensor = np.array([adj_plane for adj_plane in adj_tensor if np.any(adj_plane != all_zeros)])
        return adj_tensor


class UnitaryNoiseSampler(CircuitSampler):
    def __init__(self, noise_specs={'type':'phase_amplitude_damping_error', 'max_prob':0.1}, **kwargs):
        super(UnitaryNoiseSampler, self).__init__(**kwargs)
        self.callbacks = {'phase_amplitude_damping_error': UnitaryNoiseSampler.get_phase_amp_damp_error,
                          'amplitude_damping_error': UnitaryNoiseSampler.get_amp_damp_error}
        self.noise_specs = deepcopy(noise_specs)
        self.ops_labels = []
        self.print_verbose("Using Circuit %s" % self.circ_builder.circuit_name)
        if self.noise_specs is not None:
            self.noise = True
            self.error_type = self.noise_specs.pop('type')
            assert self.error_type in list(self.callbacks.keys()), \
                "noise 'type' must be one of: {}".format(list(self.callbacks.keys()))
            try:
                getattr(errors, self.error_type)
                self.print_verbose("Using Unitary Noise Model ({})".format(self.error_type))
            except AttributeError as e:
                print(e)
        else:
            self.noise = False
            self.print_verbose("Ideal Circuit Simulation")

    def sample(self, counts=False):
        # get a circuit + labels of unitary ops (from the circuit constructor)
        self.circ_builder.build_circuit()
        self.circuit = self.circ_builder.circuit
        self.ops_labels = self.circ_builder.ops_labels

        # build the noise model
        self.build_noise_model()
        # Either return counts
        if counts:
            return self.execute_circuit()
        # or a probability vector- this is preferred when sampling across multiple processors
        else:
            if self.noise:
                return self.get_prob_vector()
            else:
                return self.get_prob_vector(ideal=False)

    def build_noise_model(self, ideal=False):
        noise_model = NoiseModel()
        if ideal:
            self.noise_model = noise_model
            return 
        error_call = getattr(errors, self.error_type)
        for op in self.ops_labels:
            error, params = self.callbacks[self.error_type](error_call, **self.noise_specs)
            self.print_verbose(params) 
            noise_model.add_all_qubit_quantum_error(error, op)
        noise_model.add_basis_gates(['unitary'])
        self.noise_model = noise_model

    def get_prob_vector(self, ideal=True):
        """Returns the output probability vector of the sampled circuit.
        
        Keyword Arguments:
            ideal {bool} -- if True the circuit is executed twice: 1. specified noise and 2. w/o noise (default: {True})

        Returns:
            numpy.ndarray -- probability vector with shape (2**n_qubits,1) or (2**n_qubits,2) if ideal=True
        """
        noise_prob = self.execute_circuit()
        if ideal:
            old_max_prob = self.noise_specs['max_prob']
            self.noise_specs['max_prob'] = 0
            self.build_noise_model(ideal=ideal)
            ideal_prob = self.execute_circuit()
            self.noise_specs['max_prob'] = old_max_prob
            prob_arr = np.array([[noise_val, ideal_val] 
                                for (_, noise_val), (_, ideal_val) in zip(noise_prob.items(), ideal_prob.items())])
            return prob_arr
        prob_arr = np.array([noise_val for (_, noise_val) in noise_prob.items()]) 
        return prob_arr


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


if __name__ == "__main__":
    # Sample probabilities from a 5-qubit GHZ circuit with randomly inserted unitary noise channels
    noise_specs = {'type':'phase_amplitude_damping_error', 
                    'max_prob': 0.35}
    n_qubits = 5
    sampler = UnitaryNoiseSampler(noise_specs=noise_specs, verbose=False)

    # 1. Sample the circuit and get dictionary of all possible outcomes
    counts = sampler.sample(counts=True)
    # filter counts to avoid plotting 2**n_qubits
    for key in list(counts.keys()):
        if counts[key] < 5e-3:
            counts.pop(key)
    # plothistogram(counts) 

    # 2. Repeated Sampling, for each sample:
    # 2.a new random circuit is built.
    # 2.b new noise model: random unitary noise channels are inserted
    # 2.c the circuit is executed with the noise model
    for i in range(10):
        print("Sample %d:" %i)
        prob = sampler.sample() # return a numpy array 
        print("Probability Vector: ", prob)
        adjT = sampler.get_adjacency_tensor()
        print("Adjacency Tensor: ", adjT)
