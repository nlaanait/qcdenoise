from collections import OrderedDict
from typing import Dict, List
from typing import OrderedDict as OrderedDictType

import networkx as nx
import numpy as np
import qiskit.ignis.mitigation as igmit
from qiskit import QuantumCircuit
from qiskit.result.counts import Counts

from .config import get_module_logger

__all__ = ["BiSeparableWitness", "GenuineWitness"]

# module logger
logger = get_module_logger(__name__)


class Witness:
    """Base Class to estimate entanglement witnesses
    """

    def __init__(
            self,
            stabilizer_circuits: OrderedDictType[str, QuantumCircuit],
            stabilizer_counts: Dict[str, int]) -> None:
        self.stabilizer_circuits = stabilizer_circuits
        self.stabilizer_counts = stabilizer_counts
        self.diagonals = self._construct_diagonals()
        self.stabilizer_measurements = self._compute_expval()

    def _construct_diagonals(
            self) -> OrderedDictType[str, np.ndarray]:
        sgn_convert = {'-': -1., '+': 1.0}
        a = [1., 1.]
        b = [1., -1.]
        diags = OrderedDict()
        for name in self.stabilizer_circuits.keys():
            stab_sgn, stab_ops = sgn_convert[name[0]], list(name[1:])
            stab_ops_temp = stab_ops.copy()
            pop_op = stab_ops_temp.pop()
            D = a if pop_op == 'I' else b
            while len(stab_ops_temp) > 0:
                pop_op = stab_ops_temp.pop()
                d = a if pop_op == 'I' else b
                D = np.kron(d, D)
            diags[name] = np.multiply(stab_sgn, D)
        return diags

    def _compute_expval(self):
        expval = OrderedDict(
            {name: None for name in self.stabilizer_circuits.keys()})
        for idx, circ_name in enumerate(
                self.stabilizer_circuits.keys()):
            test_counts = self.stabilizer_counts[idx]
            expval[circ_name] = igmit.expectation_value(
                test_counts, self.diagonals[circ_name])
        return expval

    def estimate(self, graph: nx.Graph):
        raise NotImplementedError


class BiSeparableWitness(Witness):
    def estimate(self, graph: nx.Graph):
        pass


class GenuineWitness(Witness):
    def estimate(self, graph: nx.Graph):
        pass
