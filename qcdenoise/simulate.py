import logging
from dataclasses import dataclass
from random import sample
from typing import Union

import numpy as np
import qiskit as qk
from qiskit.providers.ibmq import IBMQBackend
from qiskit.test.mock import FakeMontreal
from qiskit.test.mock.fake_pulse_backend import FakePulseBackend
from tests.test_samplers import n_qubits


from .graph_circuit import CXGateCircuit, GraphCircuit
from .graph_data import GraphData, GraphDB, HeinGraphData
from .graph_state import GraphState
from .samplers import (CircuitSampler, DeviceNoiseSampler, NoiseSpec,
                       UnitaryNoiseSampler, HardwareSampler, unitary_noise_spec)

__all__ = [
    "GraphSpecs",
    "SamplingSpecs",
    "CircuitSpecs",
    "AdjTensorSpecs",
    "SimulationSpecs",
    "generate_adjacency_tensor",
    "sample_circuit",
    "simulate"]


# module logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    f"{__name__}- %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


@dataclass(init=True)
class GraphSpecs:
    min_num_edges: int = 2
    max_num_edges: int = 3
    directed: bool = False
    min_vertex_cover: int = 1
    max_itr: int = 1


@dataclass(init=True)
class SamplingSpecs:
    sampler: CircuitSampler
    noise_specs: NoiseSpec
    sample_options: dict
    n_shots: int = 1024


@dataclass(init=True)
class CircuitSpecs:
    builder: GraphCircuit
    stochastic: bool
    build_options: dict


@dataclass(init=True)
class AdjTensorSpecs:
    tensor_shape: tuple
    encoding: dict
    fixed_size: bool = True,
    directed: bool = False


@dataclass(init=True)
class EntanglementSpecs:
    pass


@dataclass(init=True)
class SimulationSpecs:
    n_qubits: int
    n_circuits: int
    data: GraphData
    circuit: CircuitSpecs
    backend: Union[IBMQBackend, FakePulseBackend]


# default values
graph_specs = GraphSpecs()
circuit_specs = CircuitSpecs(builder=CXGateCircuit, stochastic=True)
sample_specs = SamplingSpecs(
    sampler=UnitaryNoiseSampler,
    noise_specs=unitary_noise_spec)
adj_tensor_specs = AdjTensorSpecs(
    tensor_shape=(16, 32, 32), encoding=None)
entangle_specs = None
sim_specs = SimulationSpecs(
    n_qubits=7,
    n_circuits=10,
    circuit=circuit_specs,
    data=HeinGraphData,
    backend=FakeMontreal)


def generate_adjacency_tensor(
        dag: qk.dagcircuit.DAGCircuit,
        tensor_shape: tuple = (32, 32, 32),
        encoding: dict = None,
        fixed_size: bool = True,
        directed: bool = False) -> np.ndarray:
    """Generate an adjacency tensor representation of a circuit

    Args:
        dag (qk.dagcircuit.DAGCircuit) : directed acyclic graph returned by
        qiskit's transpiler
        tensor_shape (tuple) : dimensions of the adjacency tensor
        (# of planes, # of qubits, # of qubits)

    Keyword Args:
        encoding (dict) : encoding of gate types into integers
        fixed_size (bool) : if True returns a trimmed adjacency tensor.
        undirected (bool) : whether the adjacency tensor should be constructed
        for an undirected or directed graph.

    Returns:
        (np.ndarray) : adjacency tensor
    """
    assert isinstance(
        dag, qk.dagcircuit.DAGCircuit), \
        logging.error(
            "dag must be an instance of qiskit.dagcircuit.dagcircuit")
    adj_tensor = np.zeros(tensor_shape)
    encoding = encoding if encoding else {
        'id': 0, 'cx': 1, 'u1': 2, 'u2': 3, 'u3': 4}
    for gate in dag.gate_nodes():
        qubits = gate.qargs
        if qubits:
            if len(qubits) == 1:
                q_idx = qubits[0].index
                plane_idx = 0
                write_success = False
                while plane_idx < tensor_shape.shape[0]:
                    if adj_tensor[plane_idx, q_idx, q_idx] == 0:
                        adj_tensor[plane_idx, q_idx,
                                   q_idx] = encoding[gate.name]
                        write_success = True
                        break
                    else:
                        plane_idx += 1
                if not write_success:
                    logging.warning("max # of planes in the adjacency tensor" +
                                    "have been exceeded. Initialize a larger" +
                                    "adjacency tensor to avoid truncation.")
            if len(qubits) == 2:
                q_idx_1, q_idx_2 = [q.index for q in qubits]
                plane_idx = 0
                write_success = False
                while plane_idx < adj_tensor.shape[0]:
                    if adj_tensor[plane_idx,
                                  q_idx_1, q_idx_2] == 0:
                        adj_tensor[plane_idx, q_idx_1,
                                   q_idx_2] = encoding[gate.name]
                        if not directed:
                            adj_tensor[plane_idx, q_idx_2,
                                       q_idx_1] = encoding[gate.name]
                        write_success = True
                        break
                    else:
                        plane_idx += 1
                if not write_success:
                    logging.warning("max # of planes in the adjacency tensor" +
                                    "have been exceeded. Initialize a larger" +
                                    "adjacency tensor to avoid truncation.")

    if not fixed_size:
        # get rid of planes in adj_tensor with all id gates
        all_zeros = np.zeros_like(adj_tensor[0])
        adj_tensor = np.array(
            [adj_plane for adj_plane in adj_tensor
             if np.any(adj_plane != all_zeros)])
    return adj_tensor


def sample_circuit(graph_state: GraphState,
                   graph_specs: GraphSpecs,
                   circ_builder: GraphCircuit,
                   circ_specs: CircuitSpecs,
                   sampler: CircuitSampler,
                   sampler_specs: SamplingSpecs
                   ):
    graph = graph_state.sample(
        min_vertex_cover=graph_specs.min_vertex_cover,
        max_itr=graph_specs.max_itr)
    circ_dict = circ_builder.build(graph, **circ_specs.build_options)
    if isinstance(sampler, UnitaryNoiseSampler):
        prob_vec = sampler.sample(
            circ_dict["circuit"], circ_dict["ops"])
    elif isinstance(sampler, (DeviceNoiseSampler, HardwareSampler)):
        prob_vec = sampler.sample(
            circ_dict["circuit"],
            **sampler_specs.sample_options)
    return prob_vec, graph


def estimate_entanglement(*args, **kwargs):
    pass


def simulate(sim_specs: SimulationSpecs = sim_specs,
             sampler_specs: SamplingSpecs = sample_specs,
             graph_specs: GraphSpecs = graph_specs,
             adj_tensor_specs: AdjTensorSpecs = adj_tensor_specs,
             entangle_specs: EntanglementSpecs = None):
    # validate specs
    if sim_specs.circuit.stochastic:
        assert isinstance(sample_specs.sampler, UnitaryNoiseSampler), \
            logger.error("stochastic gates are not compatible with sampler" +
                         f"={sample_specs.sampler}, sampler shoud be set " +
                         "to UnitaryNoiseSampler")
    if isinstance(sim_specs.backend, FakePulseBackend):
        assert isinstance(sample_specs.sampler,
                          (UnitaryNoiseSampler, DeviceNoiseSampler)), \
            logger.error("Fake Backend is not compatible with sampler=" +
                         f"{sample_specs.sampler}, sampler should be one of:\n"
                         + "1. UnitaryNoiseSampler,\n2. DeviceNoiseSampler")

    # graph components
    graph_db = GraphDB(
        graph_data=sim_specs.data,
        directed=graph_specs.directed)
    graph_state = GraphState(graph_db,
                             n_qubits=sim_specs.n_qubits,
                             min_num_edges=graph_specs.min_num_edges,
                             max_num_edges=graph_specs.max_num_edges,
                             directed=graph_specs.directed)
    # circuit components
    circ_builder = sim_specs.circuit.builder(backend=sim_specs.backend,
                                             n_qubits=sim_specs.n_qubits,
                                             stochastic=sim_specs.circuit.stochastic)

    results = {}
    # sample prob vector
    if sampler_specs:
        sampler = sampler_specs.sampler(n_qubits=sim_specs.n_qubits)
        prob_vec, graph = sample_circuit(
            graph_state,
            graph_specs,
            circ_builder,
            sim_specs.circuit,
            sampler,
            sampler_specs)
        results["prob_vec"] = prob_vec
        results["graph_data"] = graph.edges.data()
    # construct adjacency tensor
    if adj_tensor_specs:
        adj_tensor = generate_adjacency_tensor(
            circ_builder.circuit_dag,
            **adj_tensor_specs)
        results["adj_tensor"] = adj_tensor

    # estimate entanglement
    if entangle_specs:
        results["entanglement"] = None
        pass
    return results
