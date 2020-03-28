import multiprocessing as mp
import os
import numpy as np
from .circuit_samplers import UnitaryNoiseSampler 
from .circuit_constructors import GHZCircuit

def sample_circuit_prob(args):
    n_samples, n_qubits, circuit_sampler, circuit_builder, noise_specs = args[:]
    sampler = circuit_sampler(n_qubits=n_qubits, noise_specs=noise_specs, 
                              circuit_builder=circuit_builder, verbose=False)
    all_prob_vec = np.empty((n_samples, 2**n_qubits, 2))
    for i in range(n_samples):
        if i%10 == 0:
            print('PID: {}, Sample #: {}'.format(os.getpid(), i))
        prob_vec = sampler.sample()
        all_prob_vec[i] = prob_vec
    return all_prob_vec

def parallel_sampler_prob(n_samples, n_qubits, circuit_sampler, circuit_builder, 
                          noise_specs, n_procs=mp.cpu_count()):
    pool = mp.Pool(n_procs, maxtasksperchild=1)
    tasks = [(n_samples, n_qubits, circuit_sampler, circuit_builder, noise_specs) 
                for _ in range(n_procs)]
    jobs = pool.map(sample_circuit_prob, tasks)
    all_results = np.concatenate([res for res in jobs])
    pool.close()
    return all_results

def sample_circuit_prob_adjT(args):
    sampling_dict = args
    noise_specs = sampling_dict.get("noise_specs", None)
    circuit_sampler = sampling_dict.get("circuit_sampler", UnitaryNoiseSampler)
    circuit_builder = sampling_dict.get("circuit_builder", GHZCircuit)
    circuit_builder_kwargs = sampling_dict.get("circuit_builder_kwargs", None)
    n_qubits = sampling_dict["n_qubits"]
    sampler = circuit_sampler(n_qubits=n_qubits, noise_specs=noise_specs, 
                              circuit_builder=circuit_builder(n_qubits=n_qubits, stochastic=True,
                              **circuit_builder_kwargs), verbose=False)
    all_prob_vec = np.empty((sampling_dict["n_samples"], 2**n_qubits, 2))
    adj_T_shape = (sampling_dict["n_samples"], sampling_dict["adjT_dim"], n_qubits, n_qubits)
    all_adj_T = np.empty(adj_T_shape)
    for i in range(sampling_dict["n_samples"]):
        if i%10 == 0:
            print('PID: {}, Sample #: {}'.format(os.getpid(), i))
        prob_vec = sampler.sample()
        all_prob_vec[i] = prob_vec
        adj_tensor = sampler.get_adjacency_tensor(max_tensor_dims=sampling_dict["adjT_dim"])
        all_adj_T[i] = adj_tensor
    return {'prob_vec':all_prob_vec, 'adj_T':all_adj_T}

def parallel_sampler_prob_adjT(sampling_dict, n_procs=mp.cpu_count()):
    pool = mp.Pool(n_procs, maxtasksperchild=1)
    # sampling_dict = [(key, itm) for key, itm in sampling_dict.items()]
    tasks = [(sampling_dict) for _ in range(n_procs)]
    # print(tasks)
    jobs = pool.map(sample_circuit_prob_adjT, tasks)
    all_results_prob = np.concatenate([res['prob_vec'] for res in jobs])
    all_results_adj_T = np.concatenate([res['adj_T'] for res in jobs])
    pool.close()
    return all_results_prob, all_results_adj_T

if __name__ == "__main__":
    # execute a q_circuit 10 times per process
    n_samples = 10
    n_procs = 4
    # 5-qubit GHZ with randomly inserted unitary operators
    noise_specs = {'type':'amplitude_damping_error', 
                    'max_prob': 0.25}
    n_qubits = 4
    circuit_sampler = UnitaryNoiseSampler
    circuit_builder = GHZCircuit
    # run it
    all_results = parallel_sampler_prob(n_samples, n_qubits, circuit_sampler, 
                                        circuit_builder, noise_specs, n_procs=n_procs)
    print(all_results.shape)
    print(all_results[0])

