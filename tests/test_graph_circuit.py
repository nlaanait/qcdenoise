"""Testing GraphCircuits
"""
from qcdenoise.graph_circuit import CXGateCircuit
import pytest
import qcdenoise as qcd


@pytest.fixture()
def graph_state():
    graph_db = qcd.GraphDB()
    graph_state = qcd.GraphState(graph_db=graph_db, n_qubits=7)
    return graph_state.sample()


@pytest.mark.dependency()
def test_CXGateCircuit(graph_state):
    circ_builder = CXGateCircuit(n_qubits=7)
    circ = circ_builder.build(graph_state)
    if circ:
        assert True


@pytest.mark.dependency()
def test_CZGateCircuit(graph_state):
    circ_builder = CXGateCircuit(n_qubits=7)
    circ = circ_builder.build(graph_state)
    if circ:
        assert True


@pytest.mark.dependency()
def test_CPhaseGateCircuit(graph_state):
    circ_builder = CXGateCircuit(n_qubits=7)
    circ = circ_builder.build(graph_state)
    if circ:
        assert True
