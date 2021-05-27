from copy import deepcopy
from typing import Dict, List, Union, OrderedDict

from collections import OrderedDict
import networkx as nx
import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit
from qiskit.providers.ibmq import IBMQBackend
from qiskit.qobj import Qobj
from qiskit.test.mock.fake_pulse_backend import FakePulseBackend
from qiskit.tools.monitor.job_monitor import job_monitor
from sympy import I
from sympy.physics.paulialgebra import Pauli, evaluate_pauli_product

from .config import get_module_logger
from .samplers import CircuitSampler, NoiseSpec

__all__ = ["TothStabilizer", "JungStabilizer"]

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

    def build(self, drop_coef: bool = True, **
              kwargs) -> OrderedDict[str, QuantumCircuit]:
        _ = self.find_stabilizers(**kwargs)
        unique_stabilizers = get_unique_operators(self.stabilizers)
        output = {sdx: None for sdx in unique_stabilizers}
        output = OrderedDict(
            {sdx: None for sdx in unique_stabilizers})
        for sdx in unique_stabilizers:
            q_reg = qk.QuantumRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, name=sdx)
            if drop_coef:
                stab_ops = list(sdx[1:])[::-1]
            else:
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
    def find_stabilizers(self, noise_robust: int = 0) -> List[str]:
        """
        noise_robust="0": no additional terms added to genuine entanglement witness
        noise_robust="1": single B vertex set
        noise_robust="2": all B vertex sets used
        """
        assert noise_robust == 0, logger.error(
            "only noise_robust=0 is currently implemented ")
        stabilizers = []
        binary_keys = [np.binary_repr(
            x, self.n_qubits) for x in range(
            2**self.n_qubits)]
        for idx in binary_keys:
            coefs = [int(x) for x in list(idx)]
            op_mat = []
            for jdx in range(len(coefs)):
                if coefs[jdx] == 0:
                    op_mat.append(
                        list('I' * self.n_qubits))
                elif coefs[jdx] == 1:
                    op_mat.append(
                        list(self.generators[jdx]))
            op_mat = np.asarray(op_mat)
            cf_arr = []
            lb_arr = []
            for kdx in range(op_mat.shape[0]):
                cf, lb = sigma_prod(
                    ''.join(op_mat[:, kdx]))
                cf_arr.append(cf)
                lb_arr.append(lb)
            if np.iscomplex(np.prod(cf_arr)):
                logger.error(
                    "Flag-error, coefficient cannot be complex")
                return
            else:
                val = np.prod(cf_arr)
                if np.real(val) == 1:
                    stabilizers.append(
                        '+' + ''.join(lb_arr))
                else:
                    stabilizers.append(
                        '-' + ''.join(lb_arr))
        self.stabilizers = deepcopy(stabilizers)
        return stabilizers

    def build(self, noise_robust: int = 0):
        return super().build(noise_robust=noise_robust)


class StabilizerSampler(CircuitSampler):
    def __init__(self, backend: Union[IBMQBackend, FakePulseBackend],
                 n_shots: int) -> None:
        super().__init__(
            backend=backend,
            n_shots=n_shots)

    def sample(self, stabilizer_circuits: Dict[str, QuantumCircuit],
               graph_circuit: QuantumCircuit, execute: bool = True):

        # concatenate graph_circuit and stabilizer_circuits
        circuits = []
        for name, stab_circuit in stabilizer_circuits.items():
            circ = graph_circuit.copy()
            stab_circuit.to_gate(label=name)
            circ.append(stab_circuit)
            circ.measure_all()
            circuits.append(circ)

        # transpile circuits
        self.transpile_circuit(circuits, {})

        results = {"counts": None, "cx_counts": None}
        if isinstance(self.backend, IBMQBackend):
            qjob_dict = self.execute_circuit(
                circuit=circuits, execute=execute)
            if execute:
                job_monitor(qjob_dict["job"], interval=5)
            results["counts"] = qjob_dict["job"].result().get_counts()
            results["cx_counts"] = self.count_cx_gates(
                stabilizer_circuits, qjob_dict["qobj"])
        else:
            results["counts"] = self.simulate_circuit(circuits)
        return results

    @staticmethod
    def count_cx_gates(stabilizer_circuits: Dict[str, QuantumCircuit],
                       q_obj: Qobj) -> Dict[str, int]:
        cx_counts = {}
        q_dict = q_obj.to_dict()
        for idx, circ_name in enumerate(stabilizer_circuits.keys()):
            cx_counter = 0
            instr = q_dict['experiments'][idx]['instructions']
            for gate in instr:
                if gate['name'] == 'cx':
                    cx_counter += 1
            cx_counts[circ_name] = cx_counter
