import pytest
import qcdenoise as qcd


@pytest.fixture()
def graph_state():
    graph_db = qcd.GraphDB()
    graph_state = qcd.GraphState(graph_db=graph_db, n_qubits=7)
    return graph_state.sample()


@pytest.mark.dependency()
def test_TothStabilizer(graph_state):
    stabilizer = qcd.TothStabilizer(graph_state, n_qubits=7)
    stab_ops = stabilizer.find_stabilizers()
    print(stab_ops)
    circuit_dict = stabilizer.build()
    print(circuit_dict)


@pytest.mark.dependency()
def test_JungStabilizer(graph_state):
    stabilizer = qcd.JungStabilizer(graph_state, n_qubits=7)
    stab_ops = stabilizer.find_stabilizers(noise_robust=1)
    print(stab_ops)
    circuit_dict = stabilizer.build()
    print(circuit_dict)
