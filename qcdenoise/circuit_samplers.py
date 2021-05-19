import datetime
import io
from copy import deepcopy
from random import choices, uniform
from warnings import warn
from typing import Tuple, List, Dict

import numpy as np
import qiskit as qk
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.providers.models.backendproperties import (BackendProperties, Gate,
                                                       Nduv)
from qiskit.quantum_info.operators import Operator

from .circuit_constructors import CircuitConstructor, GHZCircuit

global seed, dag, string
seed = 1234
np.random.seed(seed)
dag = None
string = None


def PropToNduv(props, op):
    nduv_list = []
    for prop in props:
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        name = prop["name"]
        unit = prop["unit"]
        value = prop["value"]
        nduv = Nduv(date=date, name=name, unit=unit, value=value)
        nduv_list.append(nduv)
    return nduv_list


def gatePropToGate(props, random=False):
    date = datetime.datetime.now(tz=datetime.timezone.utc)
    name = props["name"]
    gate = props["gate"]
    qubits = props["qubits"]
    parameters = PropToNduv(props["parameters"], gate)
    gate = Gate(
        gate=props["gate"],
        name=props["name"],
        parameters=parameters,
        qubits=props["qubits"])
    return gate


def get_backendProp(backend_dict):
    qbits_props = backend_dict['qubits']
    gates_props = backend_dict['gates']
    qubits_nduv = []
    for (_, qbit_prop) in enumerate(qbits_props):
        q_nduv = PropToNduv(qbit_prop, "qubit")
        qubits_nduv.append(q_nduv)
    gates = []
    for (_, gate_prop) in enumerate(gates_props):
        gate = gatePropToGate(gate_prop)
        gates.append(gate)
    date = datetime.datetime.now(tz=datetime.timezone.utc)
    general = backend_dict["general"]
    last_update_date = date
    backend_name = backend_dict["backend_name"]
    backend_version = backend_dict["backend_version"]
    backendProp = BackendProperties(backend_name=backend_name,
                                    backend_version=backend_version,
                                    qubits=qubits_nduv, gates=gates,
                                    last_update_date=last_update_date,
                                    general=general)
    return backendProp


class UnitaryNoiseSpec:
    def __init__(self):
        self.specs = {
            'type': 'phase_amplitude_damping_error',
            'max_prob': 0.1}


class DeviceNoiseSpec:
    def __init__(self):
        self.specs = {"qubit": {"T1": 0.5, "T2": 0.5, "frequency": 0.5,
                                "readout_error": 0.5, "prob_meas0_prep1": 0.5,
                                "prob_meas1_prep0": 0.5},
                      "cx": {"gate_error": 0.5, "gate_length": 0.5},
                      "id": {"gate_error": 0.5, "gate_length": 0.5},
                      "u1": {"gate_error": 0.5, "gate_length": 0.5},
                      "u2": {"gate_error": 0.5, "gate_length": 0.5},
                      "u3": {"gate_error": 0.5, "gate_length": 0.5}}


class CircuitSampler:
    def __init__(self, circuit_builder, n_qubits=2, stochastic=True, n_shots=1024,
                 verbose=True, backend=qk.Aer.get_backend('qasm_simulator'), coupling=None):
        assert n_qubits >= 2, "# of qubits must be 2 or larger"
        self.n_qubits = n_qubits
        self.circ_builder = circuit_builder
        assert isinstance(self.circ_builder, CircuitConstructor), "circuit_builder must be an instance of \
        CircuitConstructor class."
        assert self.circ_builder.n_qubits == self.n_qubits, "# of qubits is inconsistent between \
        the sampler and the circuit builder."
        self.n_shots = n_shots
        self.verbose = verbose
        self.noise_model = NoiseModel()
        self.circuit = None
        self.mapd_circuit = None
        self.gate_specs = None
        self.backend = backend
        self.coupling = coupling

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def sample(self):
        raise NotImplementedError

    def execute_circuit(self, complete_prob=True):
        job = qk.execute(self.circuit, backend=qk.Aer.get_backend('qasm_simulator'),
                         noise_model=self.noise_model, seed_simulator=seed,
                         shots=self.n_shots)
        result = job.result()
        prob_dict = result.get_counts()
        if not complete_prob:
            return prob_dict
        binary_strings = CircuitSampler.generate_binary_strings(
            self.n_qubits)
        full_prob_dict = dict([(key, 0) for key in binary_strings])
        for key, itm in prob_dict.items():
            full_prob_dict[key] = itm / self.n_shots
        return full_prob_dict

    def get_adjacency_tensor(self, max_tensor_dims=(16, 32, 32), basis_gates=['id', 'cx', 'u1', 'u2', 'u3'],
                             fixed_size=True, undirected=True, optimize=False):
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
            self.print_verbose(
                'Using noise model basis gates in transpilation.')
        if self.backend.name() == 'qasm_simulator':
            backend = None
        else:
            backend = self.backend
        # decompose() is needed to force transpiler to go into custom
        # unitary gate ops
        trans_args = {
            "backend": self.backend,
            "coupling_map": self.coupling,
            "basis_gates": basis_gates,
            "callback": get_dag}

        if optimize:
            trans_args["optimization_level"] = 3
        else:
            trans_args["optimization_level"] = 0

        self.mapd_circuit = qk.transpile(
            self.circuit.decompose(), **trans_args)
        adj_T, gate_specs = CircuitSampler.generate_adjacency_tensor(dag, adj_tensor_dim=max_tensor_dims,
                                                                     encoding=None, fixed_size=fixed_size,
                                                                     undirected=undirected)
        self.gate_specs = gate_specs
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
        bit_string = [None] * n_qubits
        # recursive call and parse string buffer
        generate_bitstring(n_qubits, bit_string, 0)
        s = string.getvalue()
        binary_strings = s.split('\n')
        binary_strings.pop()
        return binary_strings

    @staticmethod
    def generate_adjacency_tensor(
            dag, adj_tensor_dim, encoding=None, fixed_size=True, undirected=True):
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
        gate_specs = {}
        encoding = encoding if encoding else {
            'id': 0, 'cx': 1, 'u1': 2, 'u2': 3, 'u3': 4}
        assert isinstance(
            dag, qk.dagcircuit.DAGCircuit), 'dag must be an instance of qiskit.dagcircuit.dagcircuit'
        for gate in dag.gate_nodes():
            qubits = gate.qargs
            if qubits:
                if len(qubits) == 1:
                    q_idx = qubits[0].index
                    gate_name = "%s_%d" % (gate.name, q_idx)
                    if gate_name not in gate_specs.keys():
                        gate_specs[gate_name] = {
                            "gate": gate.name, "qubits": [q_idx]}
                    plane_idx = 0
                    write_success = False
                    while plane_idx < adj_tensor.shape[0]:
                        if adj_tensor[plane_idx, q_idx, q_idx] == 0:
                            adj_tensor[plane_idx, q_idx,
                                       q_idx] = encoding[gate.name]
                            write_success = True
                            break
                        else:
                            plane_idx += 1
                    if not write_success:
                        warn('max # of planes in the adjacency tensor have been exceeded. Initialize a larger \
                             adjacency tensor')
                if len(qubits) == 2:
                    q_idx_1, q_idx_2 = [q.index for q in qubits]
                    gate_name = "%s_%d_%d" % (
                        gate.name, q_idx_1, q_idx_2)
                    if gate_name not in gate_specs.keys():
                        gate_specs[gate_name] = {
                            "gate": gate.name, "qubits": [
                                q_idx_1, q_idx_2]}
                    plane_idx = 0
                    write_success = False
                    while plane_idx < adj_tensor.shape[0]:
                        if adj_tensor[plane_idx,
                                      q_idx_1, q_idx_2] == 0:
                            adj_tensor[plane_idx, q_idx_1,
                                       q_idx_2] = encoding[gate.name]
                            if undirected:
                                adj_tensor[plane_idx, q_idx_2,
                                           q_idx_1] = encoding[gate.name]
                            write_success = True
                            break
                        else:
                            plane_idx += 1
                    if not write_success:
                        warn('max # of planes in the adjacency tensor have been exceeded. Initialize a larger adjacency\
                             tensor')

        if not fixed_size:
            # get rid of planes in adj_tensor with all id gates (i.e
            # zeros)
            all_zeros = np.zeros_like(adj_tensor[0])
            adj_tensor = np.array(
                [adj_plane for adj_plane in adj_tensor if np.any(adj_plane != all_zeros)])
        return adj_tensor, gate_specs


class UnitaryNoiseSampler(CircuitSampler):
    def __init__(
            self, *args, noise_specs=UnitaryNoiseSpec().specs, **kwargs):
        super(UnitaryNoiseSampler, self).__init__(*args, **kwargs)
        self.callbacks = {'phase_amplitude_damping_error': UnitaryNoiseSampler.get_phase_amp_damp_error,
                          'amplitude_damping_error': UnitaryNoiseSampler.get_amp_damp_error}
        self.noise_specs = deepcopy(noise_specs)
        self.ops_labels = []
        self.print_verbose(
            "Using Circuit %s" %
            self.circ_builder.name)
        if self.noise_specs is not None:
            self.noise = True
            self.error_type = self.noise_specs.pop('type')
            assert self.error_type in list(self.callbacks.keys()), \
                "noise 'type' must be one of: {}".format(
                    list(self.callbacks.keys()))
            try:
                getattr(errors, self.error_type)
                self.print_verbose(
                    "Using Unitary Noise Model ({})".format(
                        self.error_type))
            except AttributeError as e:
                print(e)
        else:
            self.noise = False
            self.print_verbose("Ideal Circuit Simulation")

    def sample(self, counts=False):
        # get a circuit + labels of unitary ops (from the circuit
        # constructor)
        self.circ_builder.build_circuit()
        self.circuit = self.circ_builder.circuit
        self.ops_labels = self.circ_builder.ops_labels

        # build the noise model
        self.build_noise_model()
        # Either return counts
        if counts:
            return self.execute_circuit()
        # or a probability vector- this is preferred when sampling
        # across multiple processors
        else:
            if self.noise:
                return self.get_prob_vector()
            else:
                return self.get_prob_vector(ideal=False)

    def build_noise_model(self, ideal=False):
        noise_model = NoiseModel()
        if self.circuit is None:
            self.circ_builder.build_circuit()
            self.circuit = self.circ_builder.circuit
        if ideal:
            self.noise_model = noise_model
            return
        error_call = getattr(errors, self.error_type)
        for op in self.ops_labels:
            error, params = self.callbacks[self.error_type](
                error_call, **self.noise_specs)
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
        prob_arr = np.array(
            [noise_val for (_, noise_val) in noise_prob.items()])
        return prob_arr

    @staticmethod
    def get_phase_amp_damp_error(func, max_prob=0.1):
        phases = np.random.uniform(high=max_prob, size=2)
        amps = [min(0, 0.5 - phase)
                for phase in phases]  # hack to force CPTP
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


class DeviceNoiseSampler(CircuitSampler):
    def __init__(
            self, *args, noise_specs=DeviceNoiseSpec().specs, **kwargs):
        super(DeviceNoiseSampler, self).__init__(*args, **kwargs)
        self.backend_props = None
        self.noise_specs = deepcopy(noise_specs)
        self.print_verbose(
            "Using Circuit %s" %
            self.circ_builder.name)
        if self.noise_specs is not None:
            self.noise = True
            self.print_verbose("Using Device Noise Model")
        else:
            self.noise = False
            self.print_verbose("Ideal Circuit Simulation")

    def sample(self, counts=False):
        # get a circuit (from the circuit constructor)
        self.circ_builder.build_circuit()
        self.circuit = self.circ_builder.circuit

        # build the noise model
        self.build_noise_model()
        self.stateprep_errors()

        # Either return counts
        if counts:
            return self.execute_circuit()
        # or a probability vector- this is preferred when sampling
        # across multiple processors
        else:
            if self.noise:
                return self.get_prob_vector()
            else:
                return self.get_prob_vector(ideal=False)

    def get_prob_vector(self, ideal=True):
        """Returns the output probability vector of the sampled circuit.

        Keyword Arguments:
            ideal {bool} -- if True the circuit is executed twice: 1. specified noise and 2. w/o noise (default: {True})

        Returns:
            numpy.ndarray -- probability vector with shape (2**n_qubits,1) or (2**n_qubits,2) if ideal=True
        """
        noise_prob = self.execute_circuit()
        if ideal:
            old_noise_specs = deepcopy(self.noise_specs)
            old_noise_model = deepcopy(self.noise_model)
            old_backend_props = deepcopy(self.backend_props)
            # reduce the max possible error
            for keys, items in self.noise_specs.items():
                for key, itm in items.items():
                    self.noise_specs[keys][key] = itm / 100
            # build a new noise model
            self.build_noise_model(ideal=ideal)
            self.stateprep_errors()
            # execute
            ideal_prob = self.execute_circuit()
            self.noise_specs = old_noise_specs
            self.noise_model = old_noise_model
            self.backend_props = old_backend_props
            prob_arr = np.array([[noise_val, ideal_val]
                                 for (_, noise_val), (_, ideal_val) in zip(noise_prob.items(), ideal_prob.items())])
            return prob_arr
        prob_arr = np.array(
            [noise_val for (_, noise_val) in noise_prob.items()])
        return prob_arr

    def build_noise_model(self, ideal=False):
        noise_model = NoiseModel()
        if self.circuit is None:
            self.circ_builder.build_circuit()
            self.circuit = self.circ_builder.circuit
        if ideal:
            self.noise_model = noise_model
            return
        # call get_adjacency_tensor to populate gate_specs
        self.adjT = self.get_adjacency_tensor()
        # generate a dictionary of gate/qubit propeties and transform
        # into qiskit's BackendProperties
        self.backend_props = self.generate_backend_dict()
        backendProp = get_backendProp(self.backend_props)
        self.print_verbose("Generated BackendProps dictionary:")
        self.print_verbose(
            "qubits:\n{}\n".format(
                self.backend_props["qubits"]))
        self.print_verbose(
            "gates:\n{}\n".format(
                self.backend_props["gates"]))
        # build noise model based on backendProp
        self.noise_model = NoiseModel.from_backend(backendProp)

    def stateprep_errors(self, zero_init=True):
        q_props = self.backend_props["qubits"]
        for qbit, q_prop in zip(self.circuit.qubits, q_props):
            for error_props in q_prop:
                if error_props["name"] == "prob_meas1_prep0":
                    zero_weights = [
                        1 - error_props["value"],
                        error_props["value"]]
                elif error_props["name"] == "prob_meas0_prep1":
                    one_weights = [
                        error_props["value"],
                        1 - error_props["value"]]
            weights = zero_weights if zero_init else one_weights
            stateprep = choices([[0, 1], [1, 0]], weights=weights)[0]
            self.circuit.initialize(stateprep, qbit)

    def get_adjacency_tensor(self, max_tensor_dims=(16, 32, 32), basis_gates=['id', 'cx', 'u1', 'u2', 'u3'],
                             fixed_size=True, undirected=True, optimize=False):
        if self.circuit is None:
            self.circ_builder.build_circuit()
            self.circuit = self.circ_builder.circuit
        if self.backend_props is None:
            adj_T = super().get_adjacency_tensor(max_tensor_dims=max_tensor_dims,
                                                 basis_gates=basis_gates, fixed_size=fixed_size,
                                                 undirected=undirected, optimize=optimize)
            return adj_T
        num_qbit_planes = len(self.noise_specs["qubit"].keys())
        adj_extra = np.zeros(
            (num_qbit_planes,
             max_tensor_dims[1],
             max_tensor_dims[2]))
        for q_idx, props in enumerate(self.backend_props["qubits"]):
            idx = 0
            for prop in props:
                self.print_verbose(
                    "encoding %s at (%d, %d, %d)" %
                    (prop["name"], idx, q_idx, q_idx))
                adj_extra[idx, q_idx, q_idx] = prop["value"]
                idx += 1
        adjT = np.concatenate((adj_extra, self.adjT))
        num_gate_planes = len(self.noise_specs["cx"].keys())
        adj_extra = np.zeros(
            (num_gate_planes,
             max_tensor_dims[1],
             max_tensor_dims[2]))
        for props in self.backend_props["gates"]:
            indices = props["qubits"]
            props = props["parameters"]
            idx = 0
            for prop in props:
                if len(indices) == 1:
                    q_idx = indices[0]
                    self.print_verbose(
                        "encoding %s at (%d, %d, %d)" %
                        (prop["name"], idx, q_idx, q_idx))
                    adj_extra[idx, q_idx, q_idx] = prop["value"]
                elif len(indices) == 2:
                    q_idx_1, q_idx_2 = indices
                    self.print_verbose(
                        "encoding %s at (%d, %d, %d)" %
                        (prop["name"], idx, q_idx_1, q_idx_2))
                    adj_extra[idx, q_idx_1, q_idx_2] = prop["value"]
                    if undirected:
                        adj_extra[idx, q_idx_2,
                                  q_idx_1] = prop["value"]
                idx += 1
        adjT = np.concatenate((adj_extra, adjT))
        return adjT[:max_tensor_dims[0], :, :]

    def get_random_value(self, name, op):
        max_val = self.noise_specs[op][name]
        val = uniform(0, max_val)
        return val

    def generate_qbit_props(self):
        dicts = []
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        for name in self.noise_specs["qubit"].keys():
            if name in ["T1", "T2"]:
                value = self.get_random_value(name, "qubit")
                ent = {
                    "date": date,
                    "name": name,
                    "unit": "µs",
                    "value": value}
            else:
                value = self.get_random_value(name, "qubit")
                ent = {
                    "date": date,
                    "name": name,
                    "unit": "",
                    "value": value}
            dicts.append(ent)
        return dicts

    def generate_gate_props(self, name, specs):
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        gate_dict = {}
        gate_dict["qubits"] = specs["qubits"]
        gate_dict["gate"] = specs["gate"]
        gate_dict['name'] = name
        gate_dict["parameters"] = [{"date": date, "name": error_type, "unit": "",
                                    "value": self.get_random_value(error_type, gate_dict["gate"])}
                                   for error_type in self.noise_specs['cx'].keys()]
        return gate_dict

    def generate_backend_dict(self):
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        random_props = {"qubits": [self.generate_qbit_props() for _ in range(self.n_qubits)],
                        "gates": [self.generate_gate_props(key, spec)
                                  for key, spec in self.gate_specs.items()]}
        random_props["general"] = []
        random_props["last_update_date"] = date
        random_props["backend_name"] = self.circ_builder.name
        random_props["backend_version"] = "0.0.0"
        return random_props


class HardwareSampler(DeviceNoiseSampler):
    """Sampling of circuits on IBM's hardware backend.

    Args:
        DeviceNoiseSampler ([type]): [description]
    """

    def __init__(self, *args, **kwargs):
        """initialize
        """
        super(HardwareSampler, self).__init__(*args, **kwargs)
        assert isinstance(self.backend, qk.providers.ibmq.ibmqbackend.IBMQBackend),\
            "backend is not an instance of IBMQBackend"
        self.noise_model = None
        self.noise = False

    def screen_backend_dict(self):
        """Screen the backend properties dict and only keep keys relevant to building a noise model.

           See `DeviceNoiseSpec()` for relevant keys.
        """
        for q_idx, q_prop in enumerate(self.backend_props['qubits']):
            mod_props = [prop for prop in q_prop
                         if prop['name'] in list(self.noise_specs['qubit'].keys())]
            self.backend_props['qubits'][q_idx] = mod_props

        for g_idx, g_prop in enumerate(self.backend_props['gates']):
            gate = g_prop['gate']
            g_prop = g_prop['parameters']
            mod_props = [prop for prop in g_prop
                         if prop['name'] in list(self.noise_specs[gate].keys())]
            self.backend_props['gates'][g_idx]['parameters'] = mod_props

    def get_adjacency_tensor(self,
                             max_tensor_dims: Tuple = (16, 32, 32),
                             basis_gates: List = [
                                 'id', 'cx', 'u1', 'u2', 'u3'],
                             fixed_size=True,
                             undirected=True):
        """builds the adjacency tensor of qubits/gates after transpiling the circuit

        Args:
            max_tensor_dims (tuple, optional): adjacency tensor dimensions. Defaults to (16, 32, 32).
            basis_gates (list, optional): gates to encode in the adjacency tensor. Defaults to ['id','cx','u1','u2','u3'].
            fixed_size (bool, optional): trims the adjacency tensor to max_tensor_dims. Defaults to True.
            undirected (bool, optional): treats the circuit as undirected graph. Defaults to True.

        Returns:
            numpy.ndarray: adjacency tensor from transpiled circuit
        """
        self.adjT = super().get_adjacency_tensor(max_tensor_dims=max_tensor_dims,
                                                 basis_gates=basis_gates, fixed_size=fixed_size,
                                                 undirected=undirected, optimize=True)
        self.backend_props = self.backend.properties().to_dict()
        self.screen_backend_dict()
        return super().get_adjacency_tensor(max_tensor_dims=max_tensor_dims,
                                            basis_gates=basis_gates, fixed_size=fixed_size,
                                            undirected=undirected, optimize=True)

    def sample(self, execute=False, job_name=None):
        """sample the circuit

        Args:
            execute (bool, optional): submit circuit for execution on hardware. Defaults to False.
            job_name (string, optional): name to assign to a job. Defaults to None.

        Returns:
            dict: contains qiskit job object and ideal circuit simulation
        """
        # get a circuit (from the circuit constructor)
        self.circ_builder.build_circuit()
        self.circuit = self.circ_builder.circuit
        qjob = qk.assemble(
            self.mapd_circuit,
            shots=self.n_shots,
            backend=self.backend)
        ideal_prob = self.execute_circuit()
        if execute:
            if job_name is None:
                job_name = self.circ_builder.name
            job = self.backend.run(qjob, job_name)
            return {"job": job, "ideal_prob_dict": ideal_prob}
        return {"qjob": qjob, "ideal_prob_dict": ideal_prob}


if __name__ == "__main__":
    # Sample probabilities from a 5-qubit GHZ circuit with randomly
    # inserted unitary noise channels
    noise_specs = {'type': 'phase_amplitude_damping_error',
                   'max_prob': 0.35}
    n_qubits = 5
    sampler = UnitaryNoiseSampler(
        noise_specs=noise_specs, verbose=False)

    # 1. Sample the circuit and get dictionary of all possible
    # outcomes
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
        print("Sample %d:" % i)
        prob = sampler.sample()  # return a numpy array
        print("Probability Vector: ", prob)
        adjT = sampler.get_adjacency_tensor()
        print("Adjacency Tensor: ", adjT)
