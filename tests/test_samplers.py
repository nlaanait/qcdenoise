import pytest
import os

import qcdenoise as qcd
from qiskit.test.mock import FakeMontreal

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.fixture()
def get_circuit_dict():
    graph_db = qcd.GraphDB()
    graph_state = qcd.GraphState(graph_db=graph_db, n_qubits=7)
    state = graph_state.sample()
    circ_builder = qcd.CXGateCircuit(n_qubits=7)
    return circ_builder.build(state)


@pytest.mark.dependency()
def test_UnitaryNoiseSampler(get_circuit_dict):
    circuit_dict = get_circuit_dict
    noisy_sampler = qcd.UnitaryNoiseSampler(
        noise_specs=qcd.unitary_noise_spec)
    try:
        noisy_sampler.sample(
            circuit_dict["circuit"],
            circuit_dict["ops"])
        assert True
    except BaseException:
        pytest.fail()


@pytest.mark.dependency()
def test_DeviceNoiseSampler(get_circuit_dict):
    circuit_dict = get_circuit_dict
    noisy_sampler = qcd.DeviceNoiseSampler(
        backend=FakeMontreal,
        noise_specs=qcd.device_noise_spec)
    try:
        noisy_sampler.sample(
            circuit_dict["circuit"])
        assert True
    except BaseException:
        pytest.fail()
