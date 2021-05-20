import logging
from typing import List, Union

import networkx as nx
from networkx.classes.digraph import DiGraph
import qiskit as qk
import numpy as np
from qiskit.quantum_info.operators import Operator

__all__ = ["CXGateCircuit", "CZGateCircuit", "CPhaseGateCircuit"]

# module logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    f"{__name__}- %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class GraphCircuit:
    """Base class to construct a Quantum circuit from a Graph state
    """

    def __init__(self,
                 n_qubits: int = 3,
                 gate_type: str = None) -> None:
        """initialization is handled by this parent class

        Args:
            n_qubits (int, optional): # of qubits. Defaults to 3.
            gate_type ([type], optional): [description]. Defaults to None.
        """
        self.n_qubits = n_qubits
        self.gate_type = None
        self.circuit = None
        self.graph_iter = []

    def build(
            self, graph: Union[nx.Graph, nx.DiGraph]) -> qk.QuantumCircuit:
        """performs checks and setup of attributes needs to build a circuit.
        the circuit building functionality itself is implemented in child class
        method.

        Args:
            graph (Union[nx.Graph, nx.DiGraph]): graph state to use.

        Returns:
            qk.QuantumCircuit: quantum circuit
        """
        assert self.n_qubits == len(graph.nodes), logger.error(
            f"n_qubits={self.n_qubits} != # of graph nodes={len(graph.nodes)}")
        assert isinstance(graph, (nx.Graph, nx.DiGraph)), logger.error(
            "input graph must be of type networkx.Graph or networkx.DiGraph"
        )
        if isinstance(graph, nx.DiGraph):
            for node, ngbrs in graph.adjacency():
                for ngbr, _ in ngbrs.items():
                    self.graph_iter.append((node, ngbr))
        else:
            self.graph_iter = graph.edges()

        q_reg = qk.QuantumRegister(self.n_qubits)
        self.circuit = qk.QuantumCircuit(q_reg)


class CXGateCircuit(GraphCircuit):
    def build(
            self,
            graph: Union[nx.Graph, nx.DiGraph],
            stochastic: bool = True) -> Union[qk.QuantumCircuit, List]:
        """build a circuit from a graph state by assigning an H-CX-H gate
        sequence to each qubit and its neighbor

        Args:
            graph (Union[nx.Graph, nx.DiGraph]): graph state to use.
            stochastic (bool, optional): stochastic addition of unitary
            identity operator, which can later be processed into a Krauss
            Operator to simulate a noisy channel. Defaults to True.

        Returns:
            qk.QuantumCircuit: quantum circuit
        """
        super().build(graph)
        ops_labels = []
        self.gate_type = "CX"
        unitary_op = Operator(np.identity(4))
        self.circuit.h(range(self.n_qubits))
        for node, ngbr in self.graph_iter:
            # for ngbr, _ in ngbrs.items():
            self.circuit.h(node)
            self.circuit.cx(node, ngbr)
            # insert custom unitary after controlled gate
            if bool(np.random.choice(2)) and stochastic:
                label = f'unitary_{node}_{ngbr}'
                ops_labels.append(label)
                self.circuit.unitary(
                    unitary_op, [
                        node, ngbr], label=label)
            self.circuit.h(node)
        return {"circuit": self.circuit, "ops": ops_labels}


class CZGateCircuit(GraphCircuit):
    def build(self,
              graph: Union[nx.Graph, nx.DiGraph],
              stochastic: bool = True) -> Union[qk.QuantumCircuit, List]:
        """build a circuit from a graph state by assigning an H-CZ-H gate
        sequence to each qubit and its neighbor.

        Args:
            graph (Union[nx.Graph, nx.DiGraph]): graph state to use.
            stochastic (bool, optional): stochastic addition of unitary
            identity operator, which can later be processed into a Krauss
            Operator to simulate a noisy channel. Defaults to True.

        Returns:
            qk.QuantumCircuit: quantum circuit
        """
        super().build(graph)
        self.gate_type = "CZ"
        unitary_op = Operator(np.identity(4))
        self.circuit.h(range(self.n_qubits))
        ops_labels = []
        for node, ngbr in self.graph_iter:
            # for ngbr, _ in ngbrs.items():
            self.circuit.h(node)
            self.circuit.cz(node, ngbr)
            # insert custom unitary after controlled gate
            if bool(np.random.choice(2)) and stochastic:
                label = f'unitary_{node}_{ngbr}'
                ops_labels.append(label)
                self.circuit.unitary(
                    unitary_op, [
                        node, ngbr], label=label)
            self.circuit.h(node)
        return {"circuit": self.circuit, "ops": ops_labels}


class CPhaseGateCircuit(GraphCircuit):
    def build(self,
              graph: Union[nx.Graph, nx.DiGraph],
              Lambda: float = np.pi,
              Theta: float = np.pi / 2,
              stochastic: bool = True) -> qk.QuantumCircuit:
        """build a circuit from a graph state by assigning an Ry-CX-Ry-Cx gate
        sequence to each qubit and its neighbor.

        Args:
            graph (Union[nx.Graph, nx.DiGraph]): graph state to use.
            Theta_1 (float, optional): rotation angle for Ry.
            Defaults to np.pi.
            Theta_2 (float, optional): rotation angle for Ry.
            Defaults to np.pi/2.
            stochastic (bool, optional): stochastic addition of unitary
            identity operator, which can later be processed into a Krauss
            Operator to simulate a noisy channel. Defaults to True.

        Returns:
            qk.QuantumCircuit: quantum circuit
        """
        super().build(graph)
        self.gate_type = "CPHASE"
        unitary_op = Operator(np.identity(4))
        ops_labels = []
        for node, ngbr in self.graph_iter:
            self.circuit.ry(Lambda / 2, ngbr)
            self.circuit.cx(node, ngbr)
            if bool(np.random.choice(2)) and stochastic:
                label = f'unitary_{node}_{ngbr}'
                self.ops_labels.append(label)
                self.circuit.unitary(
                    unitary_op, [
                        node, ngbr], label=label)
            self.circuit.ry(-Theta / 2, ngbr)
            self.circuit.cx(node, ngbr)
            if bool(np.random.choice(2)) and stochastic:
                label = f'unitary_{node}_{ngbr}'
                ops_labels.append(label)
                self.circuit.unitary(
                    unitary_op, [
                        node, ngbr], label=label)
            self.circuit.ry(Lambda / 2, ngbr)
        return {"circuit": self.circuit, "ops": ops_labels}
