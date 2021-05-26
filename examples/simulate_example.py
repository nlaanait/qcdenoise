import logging

import qcdenoise as qcd
from qiskit.test.mock import FakeMontreal

logging.getLogger("qcdenoise").setLevel("INFO")

# specify simulation inputs
graph_specs = qcd.GraphSpecs(max_num_edges=10)
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
    n_qubits=11,
    n_circuits=10,
    circuit=circuit_specs,
    data=qcd.HeinGraphData,
    backend=FakeMontreal())

# call simulate
qcd.simulate(sim_specs=sim_specs,
             sampler_specs=sampler_specs,
             graph_specs=graph_specs,
             adj_tensor_specs=adj_tensor_specs,
             entangle_specs=entangle_specs)


# simulate w/o adjacency tensor
qcd.simulate(sim_specs=sim_specs,
             sampler_specs=sampler_specs,
             graph_specs=graph_specs,
             adj_tensor_specs=None,
             entangle_specs=entangle_specs)
