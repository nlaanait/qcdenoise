import numpy as np
import qcdenoise as qcd
import os, json
from datetime import datetime
os.environ["OMP_NUM_THREADS"]='1'

timestamp = datetime.now().strftime('%m%d%y')
simulate=True
data_dir = '/data/MLQC'
sampling_dict = {}
file_name = 'run_1'
train_rounds = 10
test_rounds = 1
# execute a q_circuit n times per process
n_qubits = 9
n_samples = 1000
n_procs = 38 # number of cores/processes to use
# 5-qubit GraphState Circuit with randomly inserted unitary operators
G_data = qcd.GraphData()
data_train, data_test = G_data.partition(ratio=0.1)
data_train, data_test = data_train.data, data_test.data
sampling_dict["graph_data_test"] = data_test
sampling_dict["graph_data_train"] = data_train
sampling_dict["n_samples"] = n_samples 
sampling_dict["n_qubits"] = n_qubits
sampling_dict["noise_specs"] = {"qubit":{ 
                                "readout_error":0.25, "prob_meas0_prep1": 0.1, 
                                "prob_meas1_prep0": 0.1},
                      "cx": {"gate_error": 0.25}, 
                      "id": {"gate_error": 0.25},
                      "u1": {"gate_error": 0.25},
                      "u2": {"gate_error": 0.25},
                      "u3": {"gate_error": 0.25}}
sampling_dict["circuit_sampler"] = qcd.DeviceNoiseSampler
sampling_dict["circuit_builder"] = qcd.GraphCircuit
sampling_dict["circuit_builder_kwargs"] = {"graph_data":data_train}
sampling_dict["adjT_dim"] = 16
circuit_name = 'GraphState'
sampler = 'DeviceNoise'
file_id = "{}_nqbits{}_{}_{}".format(circuit_name, n_qubits, sampler, timestamp)
data_dir = os.path.join(data_dir, file_id)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
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
            fpath = os.path.join(data_dir, 'run_prob_%s_%d.npy' %(mode, i))
            np.save(fpath, all_prob, allow_pickle=False)
            fpath = os.path.join(data_dir, 'run_adjT_%s_%d.npy' %(mode, i))
            np.save(fpath, all_adjT, allow_pickle=False)
    split = 0.9 if mode == "train" else 1
    prob_path = qcd.pool_shuffle_split(data_dir, 'run_prob_%s' % mode, mode=mode, split=split, delete=True)
    adjT_path = qcd.pool_shuffle_split(data_dir, 'run_adjT_%s' % mode, mode=mode, split=split, delete=True)
    if mode == "train":
        for mode, prob_p, adjT_p in zip(["train", "dev"], prob_path, adjT_path):
            lmdb_path = os.path.join(data_dir,'%s.lmdb' %mode)
            qcd.prob_adjT_to_lmdb(lmdb_path, prob_p, adjT_p, lmdb_map_size=int(100e9), delete=True)
    else:
        lmdb_path = os.path.join(data_dir,'%s.lmdb' %mode)
        qcd.prob_adjT_to_lmdb(lmdb_path, prob_path, adjT_path, lmdb_map_size=int(100e9), delete=True)
        
        

# dump data generation run into json
sampling_dict['sampler'] = sampler
sampling_dict['circuit_name'] = circuit_name
sampling_dict['n_samples_train'] = sampling_dict["n_samples"] * n_procs * train_rounds
sampling_dict['n_samples_test'] = sampling_dict["n_samples"] * n_procs * test_rounds
print("#### Summary #####:")
print("Executed {} Circuit using {} Sampler #{} times per process".format(sampling_dict['circuit_name'], 
                                                              sampling_dict['sampler'],
                                                              sampling_dict['n_samples']))
print("###################")

sampling_dict.pop("circuit_builder")
sampling_dict.pop("circuit_sampler")
sampling_dict.pop("circuit_builder_kwargs")
json_path = os.path.join(data_dir, 'data_specs.json')
with open(json_path, mode='w') as fp:
    json.dump(sampling_dict, fp, indent=2)
