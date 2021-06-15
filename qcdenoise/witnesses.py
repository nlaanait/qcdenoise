from collections import OrderedDict, namedtuple
from typing import Dict, List, NamedTuple
from typing import OrderedDict as OrderedDictType
from typing import Union

import networkx as nx
import numpy as np
import qiskit.ignis.mitigation as igmit
from qiskit import QuantumCircuit
from qiskit.result.counts import Counts
from uncertainties import ufloat, unumpy
from uncertainties.umath import *

from .config import get_module_logger

__all__ = ["BiSeparableWitness", "GenuineWitness"]

# module logger
logger = get_module_logger(__name__)

witness_output = namedtuple(
    'Witness', ['W_ij', 'value', 'variance'])


class Witness:
    """Base Class to estimate entanglement witnesses
    """

    def __init__(
            self,
            n_qubits: int,
            stabilizer_circuits: OrderedDictType[str, QuantumCircuit],
            stabilizer_counts: List[Counts]) -> None:
        self.n_qubits = n_qubits
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
            #stab_sgn, stab_ops = sgn_convert[name[0]], list(name[1:])
            stab_sgn, stab_ops = sgn_convert['+'], list(name)
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
    def estimate(
            self, graph: Union[nx.Graph, nx.DiGraph]) -> Dict[str, NamedTuple]:
        """ get witness of the graph state stabilizers
        """
        graph_iter = []
        if isinstance(graph, nx.DiGraph):
            for node, ngbrs in graph.adjacency():
                for ngbr, _ in ngbrs.items():
                    graph_iter.append((node, ngbr))
        else:
            graph_iter = graph.edges()

        results = {}
        for idx, edx in enumerate(graph_iter):
            e0, e1 = edx
            gi = [key for key in self.stabilizer_measurements.keys()
                  if key[::-1][e0] == 'X'][0]
            gj = [key for key in self.stabilizer_measurements.keys()
                  if key[::-1][e1] == 'X'][0]
            val = 1 - \
                self.stabilizer_measurements[gi][0] - \
                self.stabilizer_measurements[gj][0]
            meas_vals = [1.0, -
                         1. *
                         self.stabilizer_measurements[gi][0], -
                         1.0 *
                         self.stabilizer_measurements[gj][0]]
            std_vals = [
                0.0,
                self.stabilizer_measurements[gi][1],
                self.stabilizer_measurements[gj][1]]
            arr = unumpy.uarray(meas_vals, std_vals)

            results[idx] = witness_output(W_ij=tuple(
                [e0, e1], value=val, var=unumpy.std_devs(arr.sum()).tolist()))
        return results


class GenuineWitness(Witness):
    def estimate(self, graph: nx.Graph,
                 noise_robust: int = 0) -> NamedTuple:
        assert noise_robust == 0, logger.error(
            "only noise_robust=0 is implemented")
        if noise_robust == 0:
            #id_key = '+' + 'I' * self.n_qubits
            id_key = 'I' * self.n_qubits
            id_exp_val = self.stabilizer_measurements[id_key]
            id_idx = -1
            self.stabilizer_circuits.move_to_end(id_key)
            meas_vals = [
                exp_val for exp_val,
                _ in self.stabilizer_measurements.values()]
            var_vals = [
                var for _,
                var in self.stabilizer_measurements.values()]
            sum_exp_val = sum(meas_vals)
            W_ = (graph.order() - 1) * id_exp_val - \
                (sum_exp_val - id_exp_val)
            logger.info(
                "error prop. of genuine witness estimation is not implemented")
            nl = len(self.stabilizer_circuits.keys())
            ns = len(
                [key for key in self.stabilizer_circuits.keys()
                 if key != id_key])
            wit_coef = np.multiply(np.ones(nl), -1.0)
            wit_coef[id_idx] = (ns - 1)

            witness_terms = np.multiply(wit_coef, meas_vals)
            arr = unumpy.uarray(witness_terms, var_vals)
            return witness_output(
                W_ij=None, value=W_, var=unumpy.std_devs(arr.sum().tolist()))
