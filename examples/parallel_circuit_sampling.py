import numpy as np
from qcdenoise import parallel_sampler_prob, UnitaryNoiseSampler, GHZCircuit, GraphCircuit
import os, json
os.environ["OMP_NUM_THREADS"]='1'

file_name = 'run_1'
# execute a q_circuit 20 times per process
n_samples = 20
n_procs = 2 # number of cores/processes to use
# 5-qubit GraphState Circuit with randomly inserted unitary operators
noise_specs = {'type':'amplitude_damping_error', 
                'max_prob': 0.25}
n_qubits = 4
circuit_sampler = UnitaryNoiseSampler
circuit_builder = GraphCircuit

# run it. This is done in multiple rounds so that I/O is frequently flushed and spawning new processes in case some are dead.
rounds = 10
for i in range(10):
    print("Round= %d" %i)
    all_prob =parallel_sampler_prob(n_samples, n_qubits, circuit_sampler, 
                                       circuit_builder, noise_specs, n_procs=n_procs)
    np.save('run_prob_%d.npy'%i, all_prob, allow_pickle=False)

# dump data generation run into json
noise_specs['sampler'] = 'UnitaryNoise'
noise_specs['circuit_name'] = 'GraphState'
noise_specs['n_qubits'] = n_qubits
noise_specs['n_samples'] = n_samples * n_procs * rounds
print("#### Summary #####:")
print("Executed {} Circuit using {} Sampler #{} times".format(noise_specs['circuit_name'], noise_specs['sampler'],noise_specs['n_samples'] ))
print("###################")
with open('%s.json'%file_name, mode='w') as fp:
    json.dump(noise_specs, fp, indent=0)