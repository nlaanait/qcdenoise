"""Testing Simulation functions
"""
import os

import pytest
import qcdenoise as qcd
from qiskit.test.mock import FakeMontreal

os.environ["OMP_NUM_THREADS"] = "1"


@pytest.fixture()
def get_specs():
    # default values
    graph_specs = qcd.GraphSpecs()
    circuit_specs = qcd.CircuitSpecs(builder=qcd.CXGateCircuit,
                                     stochastic=True
                                     )
    sampler_specs = qcd.SamplingSpecs(
        sampler=qcd.UnitaryNoiseSampler,
        noise_specs=qcd.unitary_noise_spec)
    adj_tensor_specs = qcd.AdjTensorSpecs(
        tensor_shape=(16, 32, 32))
    entangle_specs = None
    sim_specs = qcd.SimulationSpecs(
        n_qubits=7,
        n_circuits=10,
        circuit=circuit_specs,
        data=qcd.HeinGraphData,
        backend=FakeMontreal())
    return (sim_specs, sampler_specs, graph_specs, adj_tensor_specs,
            entangle_specs)


@pytest.mark.dependency()
def test_simulate(get_specs):
    try:
        qcd.simulate(*get_specs)
        assert True
    except BaseException:
        pytest.fail()
