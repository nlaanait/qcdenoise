import os

import pytest
from qiskit.result.counts import Counts
import qcdenoise as qcd
from qiskit.test.mock import FakeMontreal

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@pytest.fixture()
def n_qubits():
    return 7


@pytest.fixture()
def graph_state(n_qubits):
    graph_db = qcd.GraphDB()
    graph_state = qcd.GraphState(graph_db=graph_db, n_qubits=n_qubits)
    return graph_state.sample()


@pytest.fixture()
def stabilizer_circuits(n_qubits, graph_state):
    stabilizer = qcd.TothStabilizer(graph_state, n_qubits=n_qubits)
    return stabilizer.build()


@pytest.fixture()
def graph_circuit(n_qubits, graph_state):
    circ_builder = qcd.CXGateCircuit(n_qubits=n_qubits)
    return circ_builder.build(graph_state)["circuit"]


@pytest.mark.dependency()
def test_TothStabilizer(graph_state):
    stabilizer = qcd.TothStabilizer(graph_state, n_qubits=7)
    stab_ops = stabilizer.find_stabilizers()
    circuit_dict = stabilizer.build()


@pytest.mark.dependency()
def test_JungStabilizer(graph_state):
    stabilizer = qcd.JungStabilizer(graph_state, n_qubits=7)
    stab_ops = stabilizer.find_stabilizers(noise_robust=0)
    circuit_dict = stabilizer.build(noise_robust=0)


@pytest.mark.dependency(depends=["test_TothStabilizer"])
def test_StabilizerSampler(stabilizer_circuits, graph_circuit):
    sampler = qcd.StabilizerSampler(
        backend=FakeMontreal(), n_shots=1024)
    counts = sampler.sample(stabilizer_circuits=stabilizer_circuits,
                            graph_circuit=graph_circuit)
    assert len(counts) == len(stabilizer_circuits.values())
    for cnt in counts:
        assert isinstance(cnt, Counts)
