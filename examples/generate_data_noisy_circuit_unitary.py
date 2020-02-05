import numpy as np
from qcdenoise import circuit_sampling_utils
import os, json
os.environ["OMP_NUM_THREADS"]='1'

file_name = 'run_1'
n_samples = 1000
n_procs = 38
# 10-qubit GHZ with randomly inserted unitary operators and noised'em up
noise_specs = {'class':'unitary_noise', 
                # 'type':'phase_amplitude_damping_error', 
                'type':'amplitude_damping_error', 
                'max_prob': 0.4, 
                'unitary_op':None}
circuit_name = 'GHZ'
n_qubits = 8
# run it
for i in range(10):
    all_results = circuit_sampling_utils.parallel_sampler(n_samples, n_qubits, circuit_name, noise_specs)
    np.save('%s_%d.npy'%(file_name, i), all_results, allow_pickle=False)

# dump data generation run into json
noise_specs['circuit_name'] = circuit_name
noise_specs['n_qubits'] = n_qubits
noise_specs['n_samples'] = n_samples * n_procs
with open('%s.json'%file_name, mode='w') as fp:
    json.dump(noise_specs, fp, indent=0)