"""Testing Simulation functions
"""
import os

import pytest
import qcdenoise as qcd
from qiskit.test.mock import FakeMontreal

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@pytest.fixture()
def sampling_specs():
    # default values
    graph_specs = qcd.GraphSpecs()
    circuit_specs = qcd.CircuitSpecs(builder=qcd.CXGateCircuit,
                                     stochastic=False
                                     )
    sampler_specs = qcd.SamplingSpecs(
        sampler=qcd.DeviceNoiseSampler,
        noise_specs=qcd.device_noise_spec)
    sim_specs = qcd.SimulationSpecs(
        n_qubits=4,
        n_circuits=10,
        circuit=circuit_specs,
        data=qcd.HeinGraphData,
        backend=FakeMontreal())
    return (sim_specs, sampler_specs, graph_specs)


@pytest.fixture()
def adj_tensor_specs():
    adj_tensor_specs = qcd.AdjTensorSpecs(
        tensor_shape=(16, 32, 32))
    return adj_tensor_specs


@pytest.fixture()
def entangle_specs():
    return qcd.EntanglementSpecs(witness=qcd.GenuineWitness,
                                 stabilizer=qcd.TothStabilizer,
                                 stabilizer_sampler=qcd.StabilizerSampler)


@pytest.mark.dependency()
def test_simulate_sampling(sampling_specs):
    try:
        qcd.simulate(*sampling_specs)
        assert True
    except BaseException:
        pytest.fail()


@pytest.mark.dependency(depends=["test_simulate_sampling"])
def test_simulate_adj_tensor(sampling_specs, adj_tensor_specs):
    try:
        qcd.simulate(
            *sampling_specs,
            adj_tensor_specs=adj_tensor_specs)
        assert True
    except BaseException:
        pytest.fail()


@pytest.mark.dependency(depends=["test_simulate_sampling"])
def test_simulate_entanglement(sampling_specs, entangle_specs):
    try:
        qcd.simulate(
            *sampling_specs,
            entangle_specs=entangle_specs)
        assert True
    except BaseException:
        pytest.fail()
