import numpy as np
import qcdenoise as qcd
import os, json
from datetime import datetime
os.environ["OMP_NUM_THREADS"]='1'

sampling_dict = {}
file_name = 'run_1'
# execute a q_circuit 20 times per process
sampling_dict["n_samples"] = 20
n_procs = 2 # number of cores/processes to use
# 5-qubit GraphState Circuit with randomly inserted unitary operators
G_data = qcd.GraphData()
data_train, data_test = G_data.partition(ratio=0.1)
data_train, data_test = data_train.data, data_test.data
sampling_dict["noise_specs"] = {'type':'amplitude_damping_error', 'max_prob': 0.25}
sampling_dict["n_qubits"] = 7
sampling_dict["circuit_sampler"] = qcd.UnitaryNoiseSampler
sampling_dict["circuit_builder"] = qcd.GraphCircuit
sampling_dict["circuit_builder_kwargs"] = {"graph_data":data_train}
sampling_dict["adjT_dim"] = 16
train_rounds = 10
test_rounds = 2
circuit_name = 'GraphState'
sampler = 'UnitaryNoise'
# run it. This is done in multiple rounds so that
# I/O is frequently flushed and spawning new processes in case some are dead.
for mode, rounds in zip(["train", "test"], [train_rounds, test_rounds]):
    print("###### Sampling %s Data ######" % mode)
    if mode == "test":
        sampling_dict["circuit_builder_kwargs"] = {"graph_data":data_test}
    for i in range(rounds):
        print("Round= %d" %i)
        all_prob, all_adjT = qcd.parallel_sampler_prob_adjT(sampling_dict, n_procs=n_procs)
        np.save('run_prob_%s_%d.npy'%(mode, i), all_prob, allow_pickle=False)
        np.save('run_adjT_%s_%d.npy'%(mode, i), all_adjT, allow_pickle=False)
    prob_path = qcd.pool_shuffle_split(os.getcwd(), 'run_prob_%s' % mode, mode=mode, split=1, delete=True)
    adjT_path = qcd.pool_shuffle_split(os.getcwd(), 'run_adjT_%s' % mode, mode=mode, split=1, delete=True)
    timestamp = datetime.now().strftime('%m%d%y')
    lmdb_path = os.path.join(os.getcwd(),'{}_nqbits{}_{}_{}_{}.lmdb'.format(circuit_name, sampling_dict["n_qubits"], sampler, timestamp, mode))
    qcd.prob_adjT_to_lmdb(lmdb_path, prob_path, adjT_path)
        

# dump data generation run into json
sampling_dict['sampler'] = sampler
sampling_dict['circuit_name'] = circuit_name
sampling_dict['n_samples_train'] = sampling_dict["n_samples"] * n_procs * train_rounds
sampling_dict['n_samples_test'] = sampling_dict["n_samples"] * n_procs * test_rounds
print("#### Summary #####:")
print("Executed {} Circuit using {} Sampler #{} times".format(sampling_dict['circuit_name'], 
                                                              sampling_dict['sampler'],
                                                              sampling_dict['n_samples']))
print("###################")

sampling_dict.pop("circuit_builder")
sampling_dict.pop("circuit_sampler")
sampling_dict.pop("circuit_builder_kwargs")
with open('%s.json'%file_name, mode='w') as fp:
    json.dump(sampling_dict, fp, indent=0)