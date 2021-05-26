import pytest
import networkx as nx
import qiskit as qk
import numpy as np


@pytest.fixture()
def n_qubits():
    return 7


@pytest.fixture()
def ghz_graph_state(n_qubits):
    edges = [(0, 1, 1)]
    for qubit in n_qubits:
        if qubit > 1:
            edge = (0, qubit, 1)
            edges.append(edge)
    return nx.Graph.add_edges_from(edges)


@pytest.fixture()
def ghz_circuit(n_qubits):
    q_reg = qk.QuantumRegister(n_qubits)
    circ = qk.QuantumCircuit(q_reg)
    circ.h(0)
    for q in range(1, n_qubits - 1):
        circ.cx(0, q + 1)
    return circ


@pytest.fixture()
def ghz_prob_vector(n_qubits):
    prob_vec = np.zeros((n_qubits,))
    prob_vec[0] = 0.5
    prob_vec[-1] = 0.5
    return prob_vec
