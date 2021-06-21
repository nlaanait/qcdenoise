from copy import deepcopy
import logging
from typing import Dict, List, Union
from typing import OrderedDict as OrderedDictType

from collections import OrderedDict
import networkx as nx
import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit
from qiskit.providers.ibmq import IBMQBackend
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.qobj import Qobj
from qiskit.result.counts import Counts
from qiskit.test.mock.fake_backend import FakeBackend
from qiskit.test.mock.fake_pulse_backend import FakePulseBackend
from qiskit.tools.monitor.job_monitor import job_monitor
from sympy import I
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product

from .config import get_module_logger
from .samplers import CircuitSampler, NoiseSpec

__all__ = ["TothStabilizer", "StabilizerSampler"]

# module logger
logger = get_module_logger(__name__)


# various helper functions

def get_unique_operators(stabilizers: list) -> list:
    """ strip leading sign +/- from stabilizer strings """
    operator_strings = [x[1:] for x in stabilizers]
    return list(set(operator_strings))


def sigma_prod(op_str: str):
    pauli_dict = {
        "I": "I",
        "X": Pauli(1),
        "Y": Pauli(2),
        "Z": Pauli(3)}
    pauli_label = {
        Pauli(1): "X",
        Pauli(2): "Y",
        Pauli(3): "Z",
        "I": "I"}
    op_list = list(op_str)
    if ('X' not in op_str) and (
            'Y' not in op_str) and ('Z' not in op_str):
        mat3 = 'I'
        coef = np.complex(1, 0)
    pauli_list = [pauli_dict[x] for x in op_list]
    coef_list = []
    while len(pauli_list) > 1:
        mat1 = pauli_list.pop()
        mat2 = pauli_list.pop()
        if mat1 == mat2:
            mat3 = 'I'
            coef = np.complex(1, 0)
        elif 'I' not in [mat1, mat2]:
            mat3 = evaluate_pauli_product(mat2 * mat1).args[-1]
            coef = evaluate_pauli_product(mat2 * mat1).args[:-1]
            if coef == (I,):
                coef = np.complex(0, 1)
            elif coef == (-1, I):
                coef = np.complex(0, -1)
            else:
                coef = np.complex(1, 0)
        else:
            mat3 = [x for x in [mat1, mat2] if x != 'I'][0]
            coef = np.complex(1, 0)
        coef_list.append(coef)
        pauli_list.append(mat3)
    # TODO: return a dict
    return np.prod(np.asarray(coef_list)), [
        pauli_label[x] for x in pauli_list][0]


def gamma_prod(gamma_ops: list):
    logger.info('WIP!!')
    raise NotImplementedError


# Main Classes


class StabilizerCircuit:
    """Base class that supports building stabilizers for graph states
    """

    def __init__(self, graph: Union[nx.Graph, nx.DiGraph],
                 n_qubits: int = 3, ) -> None:
        assert isinstance(graph, (nx.Graph, nx.DiGraph)), logger.error(
            "graph should be an instance of networkx.Graph or networkx.DiGraph")
        self.graph = graph
        self.n_qubits = n_qubits
        self.generators = self._get_generators()
        self.stabilizers = None
        self.circuits = []

    def _get_generators(self) -> list:
        """ get generators of the graph state stabilizers.
        Generators are n-length strings (n = # of qubits)
        `X` operator at vertex, `Z` operators on neighbors, `I` everywhere else.
        Generator is first constructed with leftmost index = 0
        then flipped s.t. rightmost entry corresponds to qubit 0"""
        generators = []
        if self.n_qubits > 10:
            print("not implemented for circuits with more than 10 qubits")
            raise NotImplementedError
        else:
            for idx in self.graph.nodes():
                temp = list('I' * self.n_qubits)
                temp[idx] = 'X'
                for jdx in self.graph.neighbors(idx):
                    temp[jdx] = 'Z'
                temp = "".join(temp)
                generators.append(temp[::-1])
        return generators

    def find_stabilizers(self, method: str = "Toth",
                         noise_robust: int = 0) -> List[str]:
        raise NotImplementedError

    def build(self, drop_coef: bool = False, **
              kwargs) -> OrderedDictType[str, QuantumCircuit]:
        _ = self.find_stabilizers(**kwargs)
        unique_stabilizers = get_unique_operators(self.stabilizers)
        output = {sdx: None for sdx in unique_stabilizers}
        output = OrderedDict(
            {sdx: None for sdx in unique_stabilizers})
        for sdx in unique_stabilizers:
            q_reg = qk.QuantumRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, name=sdx)
            if drop_coef:
                # drop leading character +/- and reverse Paulii string
                stab_ops = list(sdx[1:])[::-1]
            else:
                # reverse Pauili string
                stab_ops = list(sdx)[::-1]
            for idx in range(len(stab_ops)):
                op_str = stab_ops[idx]
                if op_str == 'X':
                    # measure X basis
                    circ.h(idx)
                elif op_str == 'Y':
                    # measure Y basis
                    circ.sdg(idx)
                    circ.h(idx)
                elif (op_str == 'I') or (op_str == 'Z'):
                    circ.i(idx)
            self.circuits.append(circ)
            output[sdx] = circ
        return output


class TothStabilizer(StabilizerCircuit):
    def find_stabilizers(self) -> List[str]:
        stabilizers = ['+' + x for x in self.generators]
        stabilizers.append('+' + 'I' * self.n_qubits)
        self.stabilizers = deepcopy(stabilizers)
        return stabilizers

    def build(self):
        return super().build()


class JungStabilizer(StabilizerCircuit):
    ...


class StabilizerSampler(CircuitSampler):
    def __init__(self, backend: Union[IBMQBackend, FakePulseBackend,
                                      FakeBackend],
                 n_shots: int) -> None:
        super().__init__(
            backend=backend,
            n_shots=n_shots)

    def sample(self, stabilizer_circuits: Dict[str, QuantumCircuit],
               graph_circuit: QuantumCircuit,
               execute: bool = False,
               noise_model: NoiseModel = None) -> Union[List[Counts], Qobj]:

        assert isinstance(graph_circuit, QuantumCircuit), logger.error(
            "graph_circuit should be an instance of QuantumCircuit")
        # concatenate graph_circuit and stabilizer_circuits
        circuits = []
        logging.info(
            "Adding the stabilizer circuits to the graph circuit")
        for name, stab_circuit in stabilizer_circuits.items():
            circ = graph_circuit.copy()
            stab_gate = stab_circuit.to_gate(label=name)
            circ.append(stab_gate, qargs=range(stab_gate.num_qubits))
            circ.measure_all()
            circuits.append(circ)

        # transpile circuits
        self.transpile_circuit(circuits, {})

        # build noise model
        self.build_noise_model(noise_model)

        # execute
        if isinstance(self.backend, IBMQBackend) and execute:
            qjob_dict = self.execute_circuit(
                circuit=circuits, execute=execute)
            job_monitor(qjob_dict["job"], interval=5)
            counts = qjob_dict["job"].result().get_counts()
        else:
            counts = self.simulate_circuit(circuits)
        return counts

    def build_noise_model(self, noise_model):
        if noise_model:
            self.noise_model = noise_model
            self.noisy = True
        else:
            self.noise_model = NoiseModel()
            self.noisy = False
