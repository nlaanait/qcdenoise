"""Module for generating and simulating graph state-based circuits
"""
from dataclasses import asdict, dataclass
from typing import Dict, Union

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.ibmq import IBMQBackend
from qiskit.test.mock import FakeMontreal
from qiskit.test.mock.fake_pulse_backend import FakePulseBackend

from .graph_circuit import CXGateCircuit, GraphCircuit
from .graph_data import GraphData, GraphDB, HeinGraphData
from .graph_state import GraphState
from .samplers import (CircuitSampler, DeviceNoiseSampler, HardwareSampler,
                       NoiseSpec, UnitaryNoiseSampler, encode_basis_gates,
                       generate_adjacency_tensor, unitary_noise_spec)
from .config import get_module_logger

__all__ = [
    "GraphSpecs",
    "SamplingSpecs",
    "CircuitSpecs",
    "AdjTensorSpecs",
    "SimulationSpecs",
    "adjacency_tensor_op",
    "circuit_sampling_op",
    "simulate"]


# module logger
logger = get_module_logger(__name__)


@dataclass(init=True)
class GraphSpecs:
    """Specifications for Graph State Generation
    """
    min_num_edges: int = 2
    max_num_edges: int = 3
    directed: bool = False
    min_vertex_cover: int = 1
    max_itr: int = 1


@dataclass(init=True)
class SamplingSpecs:
    """Specifications for Circuit Sampling
    """
    sampler: CircuitSampler
    noise_specs: NoiseSpec
    sample_options: dict = None
    transpile_options: dict = None
    n_shots: int = 1024


@dataclass(init=True)
class CircuitSpecs:
    """Specifications for Graph Circuit construction
    """
    builder: GraphCircuit
    stochastic: bool
    build_options: dict = None
    gate_type: str = "none"


@dataclass(init=True)
class AdjTensorSpecs:
    """Specifications for adjacency tensor representation of a circuit
    """
    tensor_shape: tuple
    fixed_size: bool = True,
    directed: bool = False


@dataclass(init=True)
class EntanglementSpecs:
    """Specifications for estimating multi-partite entanglement
    """
    pass


@dataclass(init=True)
class SimulationSpecs:
    """Specifications for carrying out simulation runs
    """
    n_qubits: int
    n_circuits: int
    data: GraphData
    circuit: CircuitSpecs
    backend: Union[IBMQBackend, FakePulseBackend]


# default values
graph_specs = GraphSpecs()
circuit_specs = CircuitSpecs(builder=CXGateCircuit,
                             stochastic=True
                             )
sample_specs = SamplingSpecs(
    sampler=UnitaryNoiseSampler,
    noise_specs=unitary_noise_spec)
adj_tensor_specs = AdjTensorSpecs(
    tensor_shape=(16, 32, 32))
entangle_specs = None
sim_specs = SimulationSpecs(
    n_qubits=7,
    n_circuits=10,
    circuit=circuit_specs,
    data=HeinGraphData,
    backend=FakeMontreal())


def adjacency_tensor_op(
        circuit: QuantumCircuit,
        backend: Union[IBMQBackend, FakePulseBackend],
        sampler: CircuitSampler,
        sampler_specs: SamplingSpecs,
        adj_tensor_specs: AdjTensorSpecs) -> np.ndarray:
    """Operation which returns an adjacency tensor. This op is wrapper around
    `qcdenoise.samplers.generate_adjacency_tensor`.

    Args:
        circuit (QuantumCircuit): quantum circuit
        backend (Union[IBMQBackend, FakePulseBackend]): quantum backend
        sampler (CircuitSampler): circuit sampler to use
        sampler_specs (SamplingSpecs): sampling specifications
        adj_tensor_specs (AdjTensorSpecs): adjacency tensor specifications

    Returns:
        np.ndarray: adjacency tensor
    """
    if sampler.circuit_dag is None:
        transpile_opts = sampler_specs.transpile_options if sampler_specs.transpile_options else {}
        sampler.transpile_circuit(
            circuit, transpile_kwargs=transpile_opts)
    adj_tensor = generate_adjacency_tensor(
        sampler.circuit_dag,
        encode_basis_gates(backend.configuration().basis_gates),
        **asdict(adj_tensor_specs))
    return adj_tensor


def circuit_sampling_op(graph_state: GraphState,
                        graph_specs: GraphSpecs,
                        circ_builder: GraphCircuit,
                        circ_specs: CircuitSpecs,
                        sampler: CircuitSampler,
                        sampler_specs: SamplingSpecs
                        ) -> Dict:
    """Operation which samples a probability vector from the quantum circuit.

    Args:
        graph_state (GraphState): generator of graph states
        graph_specs (GraphSpecs): graph generation specs
        circ_builder (GraphCircuit): circuit builder
        circ_specs (CircuitSpecs): circuit specifications
        sampler (CircuitSampler): circuit sampler
        sampler_specs (SamplingSpecs): sampling specifications

    Returns:
        Dict: `{"prob_vector": probability vector,
        "graph": nx.Graph used, "circuit": sampled quantum circuit}`
    """
    graph = graph_state.sample(
        min_vertex_cover=graph_specs.min_vertex_cover,
        max_itr=graph_specs.max_itr)
    if circ_specs.build_options:
        circ_dict = circ_builder.build(
            graph, **circ_specs.build_options)
    else:
        circ_dict = circ_builder.build(graph)
    if isinstance(sampler, UnitaryNoiseSampler):
        prob_vec = sampler.sample(
            circ_dict["circuit"], circ_dict["ops"])
    elif isinstance(sampler, (DeviceNoiseSampler, HardwareSampler)):
        sample_kwargs = sampler_specs.sample_options if sampler_specs.sample_options else {}
        prob_vec = sampler.sample(
            circ_dict["circuit"],
            **sample_kwargs)
    return {"prob_vec": prob_vec,
            "graph": graph, "circuit": circ_dict["circuit"]}


def estimate_entanglement(*args, **kwargs):
    pass


def simulate(sim_specs: SimulationSpecs,
             sampler_specs: SamplingSpecs,
             graph_specs: GraphSpecs,
             adj_tensor_specs: AdjTensorSpecs,
             entangle_specs: EntanglementSpecs = None) -> Dict:
    """Main simulation function

    Args:
        sim_specs (SimulationSpecs): simulation specs
        sampler_specs (SamplingSpecs): sampling specs
        graph_specs (GraphSpecs): graph generation specs
        adj_tensor_specs (AdjTensorSpecs): adjacency tensor specs
        entangle_specs (EntanglementSpecs, optional): entanglement estimation
        sepcs. Defaults to None.

    Returns:
        Dict: results dictionary w/ format `{"circuit_number":{"prob_vec":...,
        "graph_data":..., "adjacency_tensor":...}}`
    """
    # instantiate GraphDB, GraphState, GraphCircuit, and
    # CircuitSampler
    logger.debug(
        "Instantiating GraphDB, GraphState, GraphCircuit, CircuitSampler")
    graph_db = GraphDB(
        graph_data=sim_specs.data,
        directed=graph_specs.directed)
    graph_state = GraphState(graph_db,
                             n_qubits=sim_specs.n_qubits,
                             min_num_edges=graph_specs.min_num_edges,
                             max_num_edges=graph_specs.max_num_edges,
                             directed=graph_specs.directed)
    circ_builder = sim_specs.circuit.builder(
        n_qubits=sim_specs.n_qubits,
        stochastic=sim_specs.circuit.stochastic)
    sampler = sampler_specs.sampler(
        n_shots=sample_specs.n_shots,
        backend=sim_specs.backend,
        noise_specs=sample_specs.noise_specs)

    # validate specs
    logger.debug(
        "Validating Simulation Specs input")
    if sim_specs.circuit.stochastic:
        assert isinstance(sampler, UnitaryNoiseSampler), \
            logger.error("stochastic gates are not compatible with sampler" +
                         f"={sample_specs.sampler}, sampler shoud be set " +
                         "to UnitaryNoiseSampler")
    if isinstance(sim_specs.backend, FakePulseBackend):
        assert isinstance(sampler,
                          (UnitaryNoiseSampler, DeviceNoiseSampler)), \
            logger.error("Fake Backend is not compatible with sampler=" +
                         f"{sampler}, sampler should be one of:\n"
                         + "1. UnitaryNoiseSampler,\n2. DeviceNoiseSampler")

    results = {i: {} for i in range(sim_specs.n_circuits)}
    # sample prob vector
    for num in range(sim_specs.n_circuits):
        logger.debug(f"Simulating circuit #{num}")
        # sample prob vector
        if sampler_specs:
            logger.info("Sampling Circuit")
            sampling_op_out = circuit_sampling_op(
                graph_state,
                graph_specs,
                circ_builder,
                sim_specs.circuit,
                sampler,
                sampler_specs)
            results[num]["prob_vec"] = sampling_op_out["prob_vec"]
            results[num]["graph_data"] = list(
                sampling_op_out["graph"].edges())
        else:
            return results
        # construct adjacency tensor
        if adj_tensor_specs:
            logger.info(
                "Generating an adjacency tensor representation")
            adj_tensor = adjacency_tensor_op(
                backend=sim_specs.backend,
                circuit=sampling_op_out["circuit"],
                sampler=sampler,
                sampler_specs=sampler_specs,
                adj_tensor_specs=adj_tensor_specs)
            results[num]["adj_tensor"] = adj_tensor

        # estimate entanglement
        if entangle_specs:
            logger.info("Estimating Entanglement")
            results["entanglement"] = None
            pass
    logger.debug(f"Results of simulation: {results}")
    logger.info("Finished simulation run")
    return results
