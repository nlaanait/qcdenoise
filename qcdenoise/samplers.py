import datetime
import io
import abc
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple
from random import uniform, choices

import numpy as np
import qiskit as qk
from qiskit.providers import Backend
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.providers.models.backendproperties import (BackendProperties, Gate,
                                                       Nduv)

__all__ = ["UnitaryNoiseSampler", "unitary_noise_spec", "DeviceNoiseSampler",
           "device_noise_spec"]

# module logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    f"{__name__}- %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


# various helper functions
def generate_binary_strings(n_qubits: int) -> List[str]:
    """Generate all possible binary strings

    Args:
        n_qubits (int): # of qubits

    Returns:
        list: list of binary strings
    """
    global string
    string = io.StringIO()

    def generate_bitstring(n_qubits, bit_string, i):
        """recursively build up bitstring
        """
        if i == n_qubits:
            global string
            s = ''
            print(s.join(bit_string), file=string)
            return
        bit_string[i] = '0'
        generate_bitstring(n_qubits, bit_string, i + 1)
        bit_string[i] = '1'
        generate_bitstring(n_qubits, bit_string, i + 1)

    # recursive call and parse string buffer
    bit_string = [None] * n_qubits
    generate_bitstring(n_qubits, bit_string, 0)
    s = string.getvalue()
    binary_strings = s.split('\n')
    binary_strings.pop()
    return binary_strings


def PropToNduv(props: list) -> list:
    nduv_list = []
    for prop in props:
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        name = prop["name"]
        unit = prop["unit"]
        value = prop["value"]
        nduv = Nduv(date=date, name=name, unit=unit, value=value)
        nduv_list.append(nduv)
    return nduv_list


def gatePropToGate(props: list) -> Gate:
    gate = props["gate"]
    parameters = PropToNduv(props["parameters"])
    gate = Gate(
        gate=props["gate"],
        name=props["name"],
        parameters=parameters,
        qubits=props["qubits"])
    return gate


def get_backendProp(backend_dict: dict) -> BackendProperties:
    qbits_props = backend_dict['qubits']
    gates_props = backend_dict['gates']
    qubits_nduv = []
    for (_, qbit_prop) in enumerate(qbits_props):
        q_nduv = PropToNduv(qbit_prop)
        qubits_nduv.append(q_nduv)
    gates = []
    for (_, gate_prop) in enumerate(gates_props):
        gate = gatePropToGate(gate_prop)
        gates.append(gate)
    date = datetime.datetime.now(tz=datetime.timezone.utc)
    backendProp = BackendProperties(backend_name=backend_dict["backend_name"],
                                    backend_version=backend_dict["backend_version"],
                                    qubits=qubits_nduv,
                                    gates=gates,
                                    last_update_date=date,
                                    general=backend_dict["general"])
    return backendProp


def generate_adjacency_tensor(
        dag: qk.dagcircuit.DAGCircuit,
        adj_tensor_dim: tuple = (32, 32, 32),
        encoding: dict = None,
        fixed_size: bool = True,
        undirected: bool = True) -> Tuple[np.ndarray, dict]:
    """Generate an adjacency tensor representation of a circuit

    Args:
        dag (qk.dagcircuit.DAGCircuit) : directed acyclic graph returned by
        qiskit's transpiler
        adj_tensor_dim (tuple) : dimensions of the adjacency tensor
        (# of planes, # of qubits, # of qubits)

    Keyword Args:
        encoding (dict) : encoding of gate types into integers
        fixed_size (bool) : if True returns a trimmed adjacency tensor.
        undirected (bool) : whether the adjacency tensor should be constructed
        for an undirected or directed graph.

    Returns:
        (tuple) -- (adjacency tensor, gate_specs)
    """
    assert isinstance(
        dag, qk.dagcircuit.DAGCircuit), \
        logging.error(
            "dag must be an instance of qiskit.dagcircuit.dagcircuit")
    adj_tensor = np.zeros(adj_tensor_dim)
    gate_specs = {}
    encoding = encoding if encoding else {
        'id': 0, 'cx': 1, 'u1': 2, 'u2': 3, 'u3': 4}
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
                    logging.warning("max # of planes in the adjacency tensor" +
                                    f"have been exceeded. Initialize a larger" +
                                    f"adjacency tensor to avoid truncation.")
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
                    logging.warning("max # of planes in the adjacency tensor" +
                                    f"have been exceeded. Initialize a larger" +
                                    f"adjacency tensor to avoid truncation.")

    if not fixed_size:
        # get rid of planes in adj_tensor with all id gates
        all_zeros = np.zeros_like(adj_tensor[0])
        adj_tensor = np.array(
            [adj_plane for adj_plane in adj_tensor
             if np.any(adj_plane != all_zeros)])
    return adj_tensor, gate_specs


def convert_to_prob_vector(
        prob_dict: dict, n_qubits: int, n_shots: int):
    binary_strings = generate_binary_strings(n_qubits)
    full_prob_dict = {key: 0 for key in binary_strings}
    for key, itm in prob_dict.items():
        full_prob_dict[key] = itm / n_shots
    return np.array(
        [val for (_, val) in full_prob_dict.items()])


@dataclass(init=True)
class NoiseSpec:
    """Class to specify the noise model
    """
    specs: Dict[str, List[Tuple]] = None


# default noisespec for unitary and device noise circuit samplers
unitary_noise_spec = NoiseSpec(specs={'type': 'phase_amplitude_damping_error',
                                      'max_prob': 0.1})
device_noise_spec = NoiseSpec(specs={"qubit":
                                     {"T1": 0.5,
                                      "T2": 0.5,
                                      "frequency": 0.5,
                                      "readout_error": 0.5,
                                      "prob_meas0_prep1": 0.5,
                                      "prob_meas1_prep0": 0.5},
                                     "cx": {"gate_error": 0.5,
                                            "gate_length": 0.5},
                                     "id": {"gate_error": 0.5,
                                            "gate_length": 0.5},
                                     "u1": {"gate_error": 0.5,
                                            "gate_length": 0.5},
                                     "u2": {"gate_error": 0.5,
                                            "gate_length": 0.5},
                                     "u3": {"gate_error": 0.5,
                                            "gate_length": 0.5}})


class CircuitSampler:
    """Base class for sampling classes
    """

    def __init__(self,
                 backend: Backend = None,
                 n_shots: int = 1024,
                 noise_specs: NoiseSpec = None) -> None:
        """initialization

        Args:
            backend (Backend): quantum backend to use
            n_shots (int, optional): as it says. Defaults to 1024.
            noise_specs (NoiseSpec, optional): specs to build a noise model.
            Defaults to None.
        """
        self.n_shots = n_shots
        if backend:
            assert isinstance(backend, (Backend, abc.ABCMeta)), logger.error(
                "passed backend is not instance of qiskit.providers.Backend")
        self.backend = backend
        self.transpiled_circuit = None
        self.circuit_dag = None
        if noise_specs:
            assert isinstance(noise_specs, NoiseSpec), logger.error(
                "noise models specs is not an instance of NoiseSpec")
        self.noise_specs = noise_specs
        if self.noise_specs.specs:
            self.noisy = True
            logger.debug(
                f"Adding Noise Model with specs={self.noise_specs.specs}")
        else:
            self.noisy = False
            logger.debug("No Noise Model: ideal simulation")
        self.noise_model = None

    def sample(self, *args, **kwargs):
        """method to sample from a q-circuit

        Raises:
            NotImplementedError: must be overriden by child classes
        """
        raise NotImplementedError

    def build_noise_model(self, *args, **kwargs):
        """instantiate and populate a noise_model

        Raises:
            NotImplementedError: must be overriden by child classes
        """
        raise NotImplementedError

    def execute_circuit(self, execute: bool = False,
                        job_name: str = "placeholder"):
        if self.transpiled_circuit:
            qobj = qk.assemble(
                self.transpiled_circuit,
                shots=self.n_shots,
                backend=self.backend)
        else:
            logger.error(
                "circuit must be transpiled via CircuitSampler.transpile")
        if execute:
            job = self.backend.run(qobj, job_name)
            return {"job": job, "qobj": qobj}
        return {"job": None, "qobj": qobj}

    def simulate_circuit(self, circuit: qk.QuantumCircuit):
        logger.debug("Simulating Circuit on AerSimulator")
        job = qk.execute(circuit, backend=qk.Aer.get_backend("aer_simulator"),
                         noise_model=self.noise_model,
                         shots=self.n_shots)
        result = job.result()
        return result.get_counts(0)

    def transpile_circuit(
            self, circuit: qk.QuantumCircuit, transpile_kwargs: Dict):
        logger.debug(
            "Transpiling circuit and generating the circuit DAG")

        def get_dag(**kwargs):
            """Transpiler callable

            Returns:
                qiskit.dagcircuit.dagcircuit -- Directed Acyclic Graph
                representation of the transpiled circuit
            """
            global dag
            dag = kwargs['dag']
            return dag

        self.transpiled_circuit = qk.transpile(
            circuit.decompose(), callback=get_dag, **transpile_kwargs)
        self.circuit_dag = dag


class UnitaryNoiseSampler(CircuitSampler):
    def __init__(self,
                 n_shots: int = 1024,
                 noise_specs: NoiseSpec = None) -> None:
        super().__init__(
            n_shots=n_shots,
            noise_specs=noise_specs)

    def build_noise_model(self, ops_labels):
        self.noise_model = NoiseModel()
        error_specs = deepcopy(self.noise_specs.specs)
        error_type = error_specs.pop('type')
        error_funcs = {"phase_amplitude_damping_error":
                       UnitaryNoiseSampler.get_phase_amp_damp_error,
                       "amplitude_damping_error":
                       UnitaryNoiseSampler.get_amp_damp_error}
        if self.noisy:
            error_call = getattr(errors, error_type)
            for op in ops_labels:
                error, params = error_funcs[error_type](
                    error_call, **error_specs)
                logger.debug(f"error call parameters:{params}")
                self.noise_model.add_all_qubit_quantum_error(
                    error, op)
            self.noise_model.add_basis_gates(['unitary'])

    @ staticmethod
    def get_phase_amp_damp_error(func, max_prob=0.1):
        phases = np.random.uniform(high=max_prob, size=2)
        amps = [min(0, 0.5 - phase)
                for phase in phases]  # hack to force CPTP
        q_error_1 = func(phases[0], amps[0])
        q_error_2 = func(phases[1], amps[1])
        unitary_error = q_error_1.tensor(q_error_2)
        return unitary_error, (phases, amps)

    @ staticmethod
    def get_amp_damp_error(func, max_prob=0.1):
        amps = np.random.uniform(high=max_prob, size=2)
        q_error_1 = func(amps[0])
        q_error_2 = func(amps[1])
        unitary_error = q_error_1.tensor(q_error_2)
        return unitary_error, amps

    def sample(self, circuit: qk.QuantumCircuit,
               ops_labels: list):
        assert isinstance(circuit, qk.QuantumCircuit), logger.error(
            "passed circuit is not an instance of qiskit.QuantumCircuit")

        # attach measurement ops to circuit
        circuit.measure_all()

        # build the noise model
        self.build_noise_model(ops_labels)

        # execute circuit
        prob_dict = self.simulate_circuit(circuit)

        # generate full probability vector
        prob_vec = convert_to_prob_vector(
            prob_dict, circuit.num_qubits, self.n_shots)
        return prob_vec


class DeviceNoiseSampler(CircuitSampler):
    def __init__(self, backend: Backend = None, n_shots: int = 1024,
                 noise_specs: NoiseSpec = None) -> None:
        super().__init__(
            backend=backend,
            n_shots=n_shots,
            noise_specs=noise_specs)
        self.backend_props = None

    def sample(self, circuit: qk.QuantumCircuit):
        assert isinstance(circuit, qk.QuantumCircuit), logger.error(
            "passed circuit is not an instance of qiskit.QuantumCircuit")

        # transpile to populate circuit dag
        self.transpile_circuit(circuit, {})

        # build the noise model
        self.build_noise_model(circuit.num_qubits)

        # add state preparation errors
        self.stateprep_errors(circuit)

        # attach measurement ops to circuit
        circuit.measure_all()

        # execute circuit
        prob_dict = self.simulate_circuit(circuit)

        # generate full probability vector
        prob_vec = convert_to_prob_vector(
            prob_dict, circuit.num_qubits, self.n_shots)
        return prob_vec

    def stateprep_errors(self, circuit, zero_init=True):
        if self.noisy:
            q_props = self.backend_props["qubits"]
            for qbit, q_prop in zip(circuit.qubits, q_props):
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
                stateprep = choices(
                    [[0, 1], [1, 0]], weights=weights)[0]
                circuit.initialize(stateprep, qbit)
            logger.debug("Applied State Prep Errors")

    def build_noise_model(self, n_qubits: int):
        self.noise_model = NoiseModel()
        if self.noisy:
            # if self.backend:
            #     # populate noise_model from backend
            #     self.backend_props = self.backend().properties()
            #     self.noise_model = NoiseModel.from_backend(
            #         self.backend().properties())
            #     logger.debug(
            #         "Building NoiseModel from user input backend")
            # else:
            # construct a mock backend (properties) and use in
            # noise_model
            self.backend_props = self._generate_backend_dict(
                n_qubits)
            backendProp = get_backendProp(self.backend_props)
            logger.debug("Generated BackendProps Dict")
            logger.debug(
                f"qubits:\n{self.backend_props['qubits']}\n")
            logger.debug(
                f"gates:\n{self.backend_props['gates']}\n")
            self.noise_model = NoiseModel.from_backend(
                backendProp)
            logger.debug(
                "Building NoiseModel from custom fake backend")

    def _generate_qbit_props(self):
        dicts = []
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        for name in self.noise_specs.specs["qubit"].keys():
            if name in ["T1", "T2"]:
                value = self._get_random_value(name, "qubit")
                ent = {
                    "date": date,
                    "name": name,
                    "unit": "µs",
                    "value": value}
            else:
                value = self._get_random_value(name, "qubit")
                ent = {
                    "date": date,
                    "name": name,
                    "unit": "",
                    "value": value}
            dicts.append(ent)
        return dicts

    def _generate_gate_props(self, name, specs):
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        gate_dict = {}
        gate_dict["qubits"] = specs["qubits"]
        gate_dict["gate"] = specs["gate"]
        gate_dict['name'] = name
        gate_dict["parameters"] = [{"date": date,
                                    "name": error_type,
                                    "unit": "",
                                    "value": self._get_random_value(error_type,
                                                                    gate_dict["gate"])}
                                   for error_type in self.noise_specs.specs['cx'].keys()]
        return gate_dict

    def _generate_backend_dict(self, n_qubits: int):
        date = datetime.datetime.now(tz=datetime.timezone.utc)
        gate_specs = self._get_gate_specs()
        random_props = {"qubits": [self._generate_qbit_props()
                                   for _ in range(n_qubits)],
                        "gates": [self._generate_gate_props(key, spec)
                                  for key, spec in gate_specs.items()]}
        random_props["general"] = []
        random_props["last_update_date"] = date
        random_props["backend_name"] = "qcdenoise-fake-backend"
        random_props["backend_version"] = "0.0.0"
        return random_props

    def _get_random_value(self, name, op):
        max_val = self.noise_specs.specs[op][name]
        val = uniform(0, max_val)
        return val

    def _get_gate_specs(self):
        gate_specs = {}
        for gate in self.circuit_dag.gate_nodes():
            qubits = gate.qargs
            if qubits:
                if len(qubits) == 1:
                    q_idx = qubits[0].index
                    gate_name = "%s_%d" % (gate.name, q_idx)
                    if gate_name not in gate_specs.keys():
                        gate_specs[gate_name] = {
                            "gate": gate.name, "qubits": [q_idx]}
                elif len(qubits) == 2:
                    q_idx_1, q_idx_2 = [q.index for q in qubits]
                    gate_name = "%s_%d_%d" % (
                        gate.name, q_idx_1, q_idx_2)
                    if gate_name not in gate_specs.keys():
                        gate_specs[gate_name] = {
                            "gate": gate.name, "qubits": [
                                q_idx_1, q_idx_2]}
        return gate_specs
