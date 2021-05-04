import datetime
import multiprocessing as mp
import os

import numpy as np

from .circuit_constructors import GHZCircuit
from .circuit_samplers import UnitaryNoiseSampler


def sample_circuit_prob_adjT(args):
    sampling_dict = args
    noise_specs = sampling_dict.get("noise_specs", None)
    circuit_sampler = sampling_dict.get(
        "circuit_sampler", UnitaryNoiseSampler)
    circuit_sampler_kwargs = sampling_dict.get(
        "circuit_sampler_kwargs", {})
    circuit_builder = sampling_dict.get("circuit_builder", GHZCircuit)
    circuit_builder_kwargs = sampling_dict.get(
        "circuit_builder_kwargs", {})
    n_qubits = sampling_dict["n_qubits"]
    circ_builder = circuit_builder(n_qubits=n_qubits, stochastic=True, state_simulation=False,
                                   **circuit_builder_kwargs)
    sampler = circuit_sampler(circ_builder, n_qubits=n_qubits, noise_specs=noise_specs,
                              verbose=False, **circuit_sampler_kwargs)
    all_prob_vec = np.empty(
        (sampling_dict["n_samples"], 2**n_qubits, 2))
    adj_T_shape = (
        sampling_dict["n_samples"],
    ) + sampling_dict["adjT_dim"]
    all_adj_T = np.empty(adj_T_shape)
    for i in range(sampling_dict["n_samples"]):
        if i % 10 == 0:
            print('PID: {}, Sample #: {}'.format(os.getpid(), i))
        prob_vec = sampler.sample()
        all_prob_vec[i] = prob_vec
        adj_tensor = sampler.get_adjacency_tensor(
            max_tensor_dims=sampling_dict["adjT_dim"])
        all_adj_T[i] = adj_tensor
    return {'prob_vec': all_prob_vec, 'adj_T': all_adj_T}


def sample_circuit_prob_adjT_ew(args):
    sampling_dict = args
    noise_specs = sampling_dict.get("noise_specs", None)
    circuit_sampler = sampling_dict.get(
        "circuit_sampler", UnitaryNoiseSampler)
    circuit_sampler_kwargs = sampling_dict.get(
        "circuit_sampler_kwargs", {})
    circuit_builder = sampling_dict.get("circuit_builder", GHZCircuit)
    circuit_builder_kwargs = sampling_dict.get(
        "circuit_builder_kwargs", {})
    n_qubits = sampling_dict["n_qubits"]
    circ_builder = circuit_builder(n_qubits=n_qubits, stochastic=True, state_simulation=False,
                                   **circuit_builder_kwargs)
    sampler = circuit_sampler(circ_builder, n_qubits=n_qubits, noise_specs=noise_specs,
                              verbose=False, **circuit_sampler_kwargs)
    all_prob_vec = np.empty(
        (sampling_dict["n_samples"], 2**n_qubits, 2))
    adj_T_shape = (
        sampling_dict["n_samples"],
    ) + sampling_dict["adjT_dim"]
    all_adj_T = np.empty(adj_T_shape)
    stabilizer_measurements = []
    for i in range(sampling_dict["n_samples"]):
        if i % 10 == 0:
            print('PID: {}, Sample #: {}'.format(os.getpid(), i))
        prob_vec = sampler.sample()
        all_prob_vec[i] = prob_vec
        adj_tensor = sampler.get_adjacency_tensor(
            max_tensor_dims=sampling_dict["adjT_dim"])
        all_adj_T[i] = adj_tensor
        sampler.circ_builder.get_stabilizer_measurements(
            sampler.noise_model)
        print(sampler.circ_builder.stabilizer_measurements)
    return {'prob_vec': all_prob_vec, 'adj_T': all_adj_T}


def parallel_sampler_prob_adjT(sampling_dict, n_procs=mp.cpu_count()):
    pool = mp.Pool(n_procs, maxtasksperchild=1)
    tasks = [(sampling_dict) for _ in range(n_procs)]
    jobs = pool.map(sample_circuit_prob_adjT_ew, tasks)
    all_results_prob = np.concatenate(
        [res['prob_vec'] for res in jobs])
    all_results_adj_T = np.concatenate([res['adj_T'] for res in jobs])
    pool.close()
    return all_results_prob, all_results_adj_T


if __name__ == "__main__":
    # execute a q_circuit 10 times per process
    n_procs = 4
    sampling_dict = dict()
    sampling_dict["n_samples"] = 10
    # 5-qubit GHZ with randomly inserted unitary operators
    sampling_dict["noise_specs"] = {'type': 'amplitude_damping_error',
                                    'max_prob': 0.25}
    sampling_dict["n_qubits"] = 4
    sampling_dict["circuit_sampler"] = UnitaryNoiseSampler
    sampling_dict["circuit_builder"] = GHZCircuit
    # run it
    all_results = parallel_sampler_prob_adjT(
        sampling_dict, n_procs=n_procs)
    print(all_results.shape)
    print(all_results[0])
