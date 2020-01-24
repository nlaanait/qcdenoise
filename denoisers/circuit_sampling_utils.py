import multiprocessing as mp
import os
import numpy as np
from circuit_samplers import CircuitSampler

def sample_circuit(args):
    n_samples, n_qubits, circuit_name, noise_specs = args[:]
    circ_sampler = CircuitSampler(circuit_name=circuit_name, n_qubits=n_qubits, noise_specs=noise_specs, verbose=False)
    all_prob_vec = np.empty((n_samples, 2**n_qubits, 2))
    for i in range(n_samples):
        if i%10 == 0:
            print('PID: {}, Sample #: {}'.format(os.getpid(), i))
        circ_sampler.build_circuit()
        circ_sampler.build_noise_model()
        prob_vec = circ_sampler.get_prob_vector()
        all_prob_vec[i] = prob_vec
    return all_prob_vec


def parallel_sampler(n_samples, n_qubits, circuit_name, noise_specs, n_procs=mp.cpu_count()):
    pool = mp.Pool(n_procs, maxtasksperchild=1)
    tasks = [(n_samples, n_qubits, circuit_name, noise_specs) for _ in range(n_procs)]
    jobs = pool.map(sample_circuit, tasks)
    all_results = np.concatenate([res for res in jobs])
    pool.close()
    return all_results


if __name__ == "__main__":
    # execute a q_circuit 10 times per process
    n_samples = 10
    n_procs = 4
    # 10-qubit GHZ with randomly inserted unitary operators and noised'em up
    noise_specs = {'class':'unitary_noise', 
                    # 'type':'phase_amplitude_damping_error', 
                    'type':'amplitude_damping_error', 
                    'max_prob': 0.25, 
                    'unitary_op':None}
    circuit_name = 'GHZ'
    n_qubits = 4
    # run it
    all_results = parallel_sampler(n_samples, n_qubits, circuit_name, noise_specs)
    print(all_results.shape)
    print(all_results[0])

