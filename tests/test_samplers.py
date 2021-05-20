import os
from datetime import datetime

import pytest
import qcdenoise as qcd
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.test.mock import FakeMontreal

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.fixture()
def n_qubits():
    return 5


@pytest.fixture()
def get_hardware_backend(n_qubits):
    IBMQ.load_account()
    provider = IBMQ.get_provider(
        hub="ibm-q-ncsu",
        group="anthem",
        project="qcdenoise")
    backend = least_busy(provider.backends(
        filters=lambda x: x.configuration().n_qubits >= n_qubits and
        not x.configuration().simulator and
        x.status().operational == True))
    return backend


@pytest.fixture()
def get_circuit_sim_dict(n_qubits):
    graph_db = qcd.GraphDB()
    graph_state = qcd.GraphState(graph_db=graph_db, n_qubits=n_qubits)
    state = graph_state.sample()
    circ_builder = qcd.CXGateCircuit(n_qubits=n_qubits)
    return circ_builder.build(state)


@pytest.fixture()
def get_circuit_hw_dict(n_qubits):
    graph_db = qcd.GraphDB()
    graph_state = qcd.GraphState(graph_db=graph_db, n_qubits=n_qubits)
    state = graph_state.sample()
    circ_builder = qcd.CXGateCircuit(n_qubits=n_qubits)
    return circ_builder.build(state, stochastic=False)


@pytest.mark.dependency()
def test_UnitaryNoiseSampler(get_circuit_sim_dict):
    circuit_dict = get_circuit_sim_dict
    sampler = qcd.UnitaryNoiseSampler(
        noise_specs=qcd.unitary_noise_spec)
    try:
        sampler.sample(
            circuit_dict["circuit"],
            circuit_dict["ops"])
        assert True
    except BaseException:
        pytest.fail()


@pytest.mark.dependency()
def test_DeviceNoiseSampler(get_circuit_sim_dict):
    circuit_dict = get_circuit_sim_dict
    sampler = qcd.DeviceNoiseSampler(
        backend=FakeMontreal(),
        noise_specs=qcd.device_noise_spec)
    try:
        sampler.sample(
            circuit_dict["circuit"])
        assert True
    except BaseException:
        pytest.fail()


@pytest.mark.dependency()
def test_HardwareSampler(get_circuit_hw_dict, get_hardware_backend):
    circuit_dict = get_circuit_hw_dict
    sampler = qcd.HardwareSampler(get_hardware_backend, n_shots=1024)
    timestamp = datetime.now()
    job_name = f"qcdenoise-test-{timestamp}"
    try:
        sampler.sample(
            circuit_dict["circuit"],
            execute=True,
            job_name=job_name)
        assert True
    except BaseException:
        pytest.fail()
