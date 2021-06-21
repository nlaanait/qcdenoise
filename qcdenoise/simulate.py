"""Module for generating and simulating graph state-based circuits
"""
from copy import deepcopy
from dataclasses import asdict, dataclass

from qcdenoise.stabilizers import StabilizerSampler, TothStabilizer
from typing import Dict, NamedTuple, Union

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.providers.ibmq import IBMQBackend
from qiskit.test.mock import FakeMontreal, FakeBackend
from qiskit.test.mock.fake_pulse_backend import FakePulseBackend
import networkx as nx

from .config import get_module_logger
from .graph_circuit import (CPhaseGateCircuit, CXGateCircuit, CZGateCircuit,
                            GraphCircuit)
from .graph_data import GraphData, GraphDB, HeinGraphData
from .graph_state import GraphState
from .samplers import (DeviceNoiseSampler, HardwareSampler, NoiseSpec,
                       UnitaryNoiseSampler, encode_basis_gates,
                       generate_adjacency_tensor, unitary_noise_spec)
from .witnesses import BiSeparableWitness, GenuineWitness

__all__ = [
    "GraphSpecs",
    "SamplingSpecs",
    "CircuitSpecs",
    "AdjTensorSpecs",
    "SimulationSpecs",
    "EntanglementSpecs",
    "adjacency_tensor_op",
    "circuit_sampling_op",
    "estimate_entanglement_op",
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
    sampler: Union[DeviceNoiseSampler, UnitaryNoiseSampler]
    noise_specs: NoiseSpec
    sample_options: dict = None
    transpile_options: dict = None
    n_shots: int = 1024


@dataclass(init=True)
class CircuitSpecs:
    """Specifications for Graph Circuit construction
    """
    builder: Union[CXGateCircuit, CZGateCircuit, CPhaseGateCircuit]
    stochastic: bool
    build_options: dict = None
    gate_type: str = "none"


@dataclass(init=True)
class AdjTensorSpecs:
    """Specifications for adjacency tensor representation of a circuit
    """
    tensor_shape: tuple
    fixed_size: bool = True
    directed: bool = False


@dataclass(init=True)
class EntanglementSpecs:
    """Specifications for estimating multi-partite entanglement
    """
    witness: Union[GenuineWitness, BiSeparableWitness]
    stabilizer: TothStabilizer
    stabilizer_sampler: StabilizerSampler = StabilizerSampler
    n_shots: int = 1024


@dataclass(init=True)
class SimulationSpecs:
    """Specifications for carrying out simulation runs
    """
    n_qubits: int
    n_circuits: int
    data: GraphData
    circuit: CircuitSpecs
    backend: Union[IBMQBackend, FakePulseBackend] = None


def adjacency_tensor_op(
        circuit: QuantumCircuit,
        backend: Union[IBMQBackend, FakePulseBackend],
        sampler: Union[UnitaryNoiseSampler,
                       DeviceNoiseSampler,
                       HardwareSampler],
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


def circuit_sampling_op(graph: Union[nx.Graph, nx.DiGraph],
                        circ_builder: GraphCircuit,
                        circ_specs: CircuitSpecs,
                        sampler: Union[DeviceNoiseSampler,
                                       UnitaryNoiseSampler,
                                       HardwareSampler],
                        sampler_specs: SamplingSpecs
                        ) -> Dict:
    """Operation which samples a probability vector from the quantum circuit.

    Args:
        graph (nx.Graph, nx.Digraph): graph state
        circ_builder (GraphCircuit): circuit builder
        circ_specs (CircuitSpecs): circuit specifications
        sampler (CircuitSampler): circuit sampler
        sampler_specs (SamplingSpecs): sampling specifications

    Returns:
        Dict: `{"prob_vector": probability vector,
        "graph": nx.Graph used, "circuit": sampled quantum circuit}`
    """
    # graph = graph_state.sample(
    #     min_vertex_cover=graph_specs.min_vertex_cover,
    #     max_itr=graph_specs.max_itr)
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
            "graph": graph,
            "circuit": circ_dict["circuit"],
            "noise_model": deepcopy(sampler.noise_model)}


def estimate_entanglement_op(
        graph: Union[nx.Graph, nx.DiGraph],
        graph_circuit: QuantumCircuit,
        entangle_specs: EntanglementSpecs,
        n_qubits: int,
        backend: Union[IBMQBackend,
                       FakeBackend,
                       FakePulseBackend] = None) -> NamedTuple:
    stabilizer = entangle_specs.stabilizer(graph, n_qubits)
    stab_circs = stabilizer.build()
    sampler = entangle_specs.stabilizer_sampler(
        n_shots=entangle_specs.n_shots, backend=backend)
    counts = sampler.sample(
        stabilizer_circuits=stab_circs,
        graph_circuit=graph_circuit)
    logger.info(f"counts={counts}")
    witness = entangle_specs.witness(
        n_qubits=n_qubits,
        stabilizer_circuits=stab_circs,
        stabilizer_counts=counts)
    return witness.estimate(graph=graph)


def simulate(sim_specs: SimulationSpecs,
             sampler_specs: SamplingSpecs,
             graph_specs: GraphSpecs,
             adj_tensor_specs: AdjTensorSpecs = None,
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
    # results dictionary
    results = {i: {} for i in range(sim_specs.n_circuits)}

    # validate inputs
    logger.debug("Validating Input")
    assert isinstance(sim_specs, SimulationSpecs), logger.error(
        "sim_specs must be an instance of SimulationSpecs")
    assert isinstance(sampler_specs, SamplingSpecs), logger.error(
        "sampler_specs must be an instance of SamplingSpecs")
    assert isinstance(graph_specs, GraphSpecs), logger.error(
        "graph_specs must be an instance of SimulationSpecs")
    if adj_tensor_specs:
        assert isinstance(adj_tensor_specs, AdjTensorSpecs), logger.error(
            "adj_tensor_specs must be an instance of AdjTensorSpecs")
    if entangle_specs:
        assert isinstance(entangle_specs, EntanglementSpecs), logger.error(
            "entangle_specs must be an instance of EntanglementSpecs")

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
        n_shots=sampler_specs.n_shots,
        backend=sim_specs.backend,
        noise_specs=sampler_specs.noise_specs)

    # validate specs
    logger.debug(
        "Validating Simulation Specs input")
    if sim_specs.circuit.stochastic:
        assert isinstance(sampler, UnitaryNoiseSampler), \
            logger.error("stochastic gates are not compatible with sampler" +
                         f"={sampler_specs.sampler}, sampler shoud be set " +
                         "to UnitaryNoiseSampler")
    if isinstance(sim_specs.backend, (FakeBackend, FakePulseBackend)):
        assert isinstance(sampler,
                          (UnitaryNoiseSampler, DeviceNoiseSampler)), \
            logger.error("Fake Backend is not compatible with sampler=" +
                         f"{sampler}, sampler should be one of:\n"
                         + "1. UnitaryNoiseSampler,\n2. DeviceNoiseSampler")

    # Simulating
    for num in range(sim_specs.n_circuits):
        # sample graph state
        logger.info("Sampling Graph State")
        logger.debug(f"iter: {num}- Sampling a graph state")
        graph = graph_state.sample(
            min_vertex_cover=graph_specs.min_vertex_cover,
            max_itr=graph_specs.max_itr)
        results[num]["graph_data"] = list(graph.edges())
        logger.debug(f"iter: {num}- Simulating circuit")

        # sample prob vector
        logger.info("Building Circuit and Measurement")
        sampling_op_out = circuit_sampling_op(
            graph,
            circ_builder,
            sim_specs.circuit,
            sampler,
            sampler_specs)
        results[num]["prob_vec"] = sampling_op_out["prob_vec"]

        # construct adjacency tensor
        logger.debug(f"iter: {num}- Building Adjacency Tensor")
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
        logger.debug(f"iter: {num}- Estimating Entanglement")
        logger.info(
            "Estimating Multi-partite Entanglement")
        if entangle_specs:
            entangle_op_out = estimate_entanglement_op(
                graph=graph,
                entangle_specs=entangle_specs,
                n_qubits=sim_specs.n_qubits,
                graph_circuit=sampling_op_out["circuit"],
                backend=sim_specs.backend)

            results[num]["entanglement"] = {"value": entangle_op_out.value,
                                            "variance": entangle_op_out.variance}

    logger.debug(f"Results of simulation: {results}")
    logger.info("Finished simulation run")
    return results
