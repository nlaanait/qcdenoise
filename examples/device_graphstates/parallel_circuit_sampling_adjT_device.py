import numpy as np
import qcdenoise as qcd
import os, json
from datetime import datetime
os.environ["OMP_NUM_THREADS"]='1'

simulate=True

sampling_dict = {}
file_name = 'run_1'
# execute a q_circuit 20 times per process
sampling_dict["n_samples"] = 1000
n_procs = 38 # number of cores/processes to use
# 5-qubit GraphState Circuit with randomly inserted unitary operators
G_data = qcd.GraphData()
data_train, data_test = G_data.partition(ratio=0.1)
data_train, data_test = data_train.data, data_test.data
sampling_dict["noise_specs"] = {"qubit":{ 
                                "readout_error":0.5, "prob_meas0_prep1": 0.5, 
                                "prob_meas1_prep0": 0.5},
                      "cx": {"gate_error": 0.5, "gate_length": 0.5}, 
                      "id": {"gate_error": 0.5, "gate_length": 0.5},
                      "u1": {"gate_error": 0.5, "gate_length": 0.5},
                      "u2": {"gate_error": 0.5, "gate_length": 0.5},
                      "u3": {"gate_error": 0.5, "gate_length": 0.5}}  
sampling_dict["n_qubits"] = 9
sampling_dict["circuit_sampler"] = qcd.DeviceNoiseSampler
sampling_dict["circuit_builder"] = qcd.GraphCircuit
sampling_dict["circuit_builder_kwargs"] = {"graph_data":data_train}
sampling_dict["adjT_dim"] = 16
train_rounds = 20
test_rounds = 1
circuit_name = 'GraphState'
sampler = 'DeviceNoise'
data_dir = '/data/MLQC'
# run it. This is done in multiple rounds so that
# I/O is frequently flushed and spawning new processes in case some are dead.
for mode, rounds in zip(["train", "test"], [train_rounds, test_rounds]):
    print("###### Sampling %s Data ######" % mode)
    if mode == "test":
        sampling_dict["circuit_builder_kwargs"] = {"graph_data":data_test}
    if simulate:
        for i in range(rounds):
            print("Round= %d" %i)
            all_prob, all_adjT = qcd.parallel_sampler_prob_adjT(sampling_dict, n_procs=n_procs)
            np.save('%s/run_prob_%s_%d.npy'%(data_dir, mode, i), all_prob, allow_pickle=False)
            np.save('%s/run_adjT_%s_%d.npy'%(data_dir, mode, i), all_adjT, allow_pickle=False)
    prob_path = qcd.pool_shuffle_split(data_dir, 'run_prob_%s' % mode, mode=mode, split=1, delete=False)
    adjT_path = qcd.pool_shuffle_split(data_dir, 'run_adjT_%s' % mode, mode=mode, split=1, delete=False)
    timestamp = datetime.now().strftime('%m%d%y')
    lmdb_path = os.path.join(data_dir,'{}_nqbits{}_{}_{}_{}.lmdb'.format(circuit_name, sampling_dict["n_qubits"], sampler, timestamp, mode))
    qcd.prob_adjT_to_lmdb(lmdb_path, prob_path, adjT_path, lmdb_map_size=int(100e9), delete=False)
        

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
