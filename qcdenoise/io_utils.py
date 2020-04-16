import os
import random
import shlex
import subprocess
from time import time

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

#pylint: disable=no-member
global seed
seed = 1234
np.random.seed(seed)
random.seed(seed)

def pool_shuffle_split(files_dir, file_expr, mode='train', split=0.8, delete=True):
    files = os.listdir(files_dir)
    files = [file for file in files if file_expr in file]
    runs = np.concatenate([np.load(os.path.join(files_dir, file), mmap_mode='r') for file in files])
    for _ in range(5):
        np.random.shuffle(runs)
    if split > 0.999:
        path = os.path.join(files_dir, file_expr+'.npy')
        np.save(path, runs, allow_pickle=False)
        print('wrote {} with shape {}'.format(path, runs.shape))
        if delete:
            for file in files:
                file_path = os.path.join(files_dir, file)
                args = "rm %s" %file_path
                args = shlex.split(args)
                if os.path.exists(file_path):
                    try:
                        subprocess.run(args, check=True, timeout=120)
                        print("rm %s" % file_path)
                    except subprocess.SubprocessError as e:
                        print("failed to rm %s" % file_path)
                        print(e) 
        return path 
    else:
        part = int(runs.shape[0] * split)
        train = runs[:part]
        test = runs[part:]
        path_train = os.path.join(files_dir, file_expr+'_train.npy') 
        np.save(path_train, train, allow_pickle=False)
        print('wrote {} with shape {}'.format(path_train, train.shape))
        path_dev = os.path.join(files_dir, file_expr+'_dev.npy') 
        np.save(path_dev, test, allow_pickle=False)
        print('wrote {} with shape {}'.format(path_dev, test.shape))
    cond = os.path.exists(path_dev) and os.path.exists(path_train)
    if delete and cond:
        for file in files:
            file_path = os.path.join(files_dir, file)
            args = "rm %s" %file_path
            args = shlex.split(args)
            if os.path.exists(file_path):
                try:
                    subprocess.run(args, check=True, timeout=120)
                    print("rm %s" % file_path)
                except subprocess.SubprocessError as e:
                    print("failed to rm %s" % file_path)
                    print(e)
    return path_train, path_dev

def prob_adjT_to_lmdb(lmdb_path, prob_data_path, adjT_data_path, lmdb_map_size=int(10e9), delete=True):
    env = lmdb.open(lmdb_path, map_size=lmdb_map_size, map_async=True, writemap=True, create=True)
    prob_data = np.load(prob_data_path, mmap_mode='r')
    adjT_data = np.load(adjT_data_path, mmap_mode='r')
    with env.begin(write=True) as txn:
        for (idx, prob) , adjT in zip(enumerate(prob_data), adjT_data):
            prob_noise = prob[:,0]
            prob_ideal = prob[:,1]
            inputs_shape = prob_noise.shape
            targets_shape = prob_ideal.shape
            encoding_shape = adjT.shape
            inputs = prob_noise.flatten().tostring()
            key = bytes('input_%s'%format(idx), "ascii")
            txn.put(key, inputs)
            key = bytes('target_%s'%format(idx), "ascii")
            targets = prob_ideal.flatten().tostring()
            txn.put(key, targets)
            key = bytes('encoding_%s'%format(idx), "ascii")
            encodings = adjT.flatten().tostring()
            txn.put(key, encodings)

        env.sync()

        headers = { b"input_dtype": bytes(prob_noise.dtype.str, "ascii"),
                    b"input_shape": np.array(inputs_shape).tostring(),
                    b"target_shape": np.array(targets_shape).tostring(),
                    b"target_dtype": bytes(prob_ideal.dtype.str, "ascii"),
                    b"encoding_shape": np.array(encoding_shape).tostring(),
                    b"encoding_dtype": bytes(adjT.dtype.str, "ascii"), 
                    b"target_name": bytes('target_', "ascii"),
                    b"input_name": bytes('input_', "ascii"),
                    b"encoding_name": bytes('encoding_', "ascii")}
        for key, val in headers.items():
            txn.put(key, val)
        txn.put(b"header_entries", bytes(len(list(headers.items()))))
        txn.put(b"num_samples", bytes('%s' %str(idx+1), "ascii"))
        env.sync()
    print('wrote lmdb database in %s' % lmdb_path)
    if delete:
        for file_path in [prob_data_path, adjT_data_path]:
            args = "rm %s" %file_path
            args = shlex.split(args)
            if os.path.exists(file_path):
                try:
                    subprocess.run(args, check=True, timeout=120)
                    print("rm %s" % file_path)
                except subprocess.SubprocessError as e:
                    print("failed to rm %s" % file_path)
                    print(e)

class QCIRCDataSet(Dataset):
    """ QCIRC data set on lmdb."""
    def __init__(self, lmdb_path, input_transform=None, target_transform=None,
                                        debug=False):
        """__init__ [summary]
        
        Args:
            lmdb_path ([type]): [description]
            key_base (str, optional): [description]. Defaults to 'sample'.
            input_transform ([type], optional): [description]. Defaults to QCIRCTransform().
            target_transform ([type], optional): [description]. Defaults to None.
            debug (bool, optional): [description]. Defaults to True.
        """
        self.debug = debug
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(self.lmdb_path, create=False, readahead=False, readonly=True, writemap=False, lock=False)
        with self.env.begin(write=False) as txn:
            input_shape = np.frombuffer(txn.get(b"input_shape"), dtype='int64')
            target_shape = np.frombuffer(txn.get(b"target_shape"), dtype='int64')
            encoding_shape = np.frombuffer(txn.get(b"encoding_shape"), dtype='int64')
            input_dtype = np.dtype(txn.get(b"input_dtype").decode("ascii"))
            target_dtype = np.dtype(txn.get(b"target_dtype").decode("ascii"))
            encoding_dtype = np.dtype(txn.get(b"encoding_dtype").decode("ascii"))
            target_name = txn.get(b"target_name").decode("ascii")
            input_name = txn.get(b"input_name").decode("ascii")
            encoding_name = txn.get(b"encoding_name").decode("ascii")
            self.num_samples = int(txn.get(b"num_samples").decode("ascii"))
        self.records = np.arange(0, self.num_samples, dtype=np.int) 
        for _ in range(10):
            np.random.shuffle(self.records)
        self.data_specs={'input_shape': list(input_shape), 'target_shape': list(target_shape), 'encoding_shape': list(encoding_shape),
                         'target_dtype':target_dtype, 'input_dtype': input_dtype, 'encoding_dtype': encoding_dtype,
                         'target_key':target_name, 'input_key': input_name, 'encoding_key': encoding_name}
        self.input_keys = [bytes(self.data_specs['input_key']+str(idx), "ascii") for idx in self.records]
        self.target_keys = [bytes(self.data_specs['target_key']+str(idx), "ascii") for idx in self.records]
        self.encoding_keys = [bytes(self.data_specs['encoding_key']+str(idx), "ascii") for idx in self.records]
        self.print_debug("Opened lmdb file %s, with %d samples" %(self.lmdb_path, self.num_samples))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # outside_func(idx)
        input_key = self.input_keys[idx]
        target_key = self.target_keys[idx]
        encoding_key = self.encoding_keys[idx]
        with self.env.begin(write=False, buffers=True) as txn:
            input_bytes = txn.get(input_key)
            target_bytes = txn.get(target_key)
            encoding_bytes = txn.get(encoding_key)
        inputs = np.frombuffer(input_bytes, dtype=self.data_specs['input_dtype'])
        inputs = inputs.reshape(self.data_specs['input_shape'])
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        targets = np.frombuffer(target_bytes, dtype=self.data_specs['target_dtype'])
        targets = targets.reshape(self.data_specs['target_shape'])
        if targets.dtype != np.float32:
            targets = targets.astype(np.float32)
        encodings = np.frombuffer(encoding_bytes, dtype=self.data_specs['encoding_dtype'])
        encodings = encodings.reshape(self.data_specs['encoding_shape'])
        if encodings.dtype != np.float32:
            encodings = encodings.astype(np.float32)
        self.print_debug('read (inputs, targets, encodings) # {idx} with shape ({in_shape}, {trg_shape}, {enc_shape})'\
                        .format(idx=idx, in_shape=inputs.shape, trg_shape=targets.shape, enc_shape=encodings.shape))
        if self.input_transform is not None:
            inputs = self.transform_input(inputs)
        if self.target_transform is not None:
            targets = self.transform_target(targets)
        if False: # this is only needed if we actually use transforms
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.from_numpy(inputs) 
            if not isinstance(targets, torch.Tensor):
                targets = torch.from_numpy(targets)
            if not isinstance(encodings, torch.Tensor):
                encodings = torch.from_numpy(encodings)
            return {'input':inputs, 'target':targets, 'encoding': encodings}
        return {'input':torch.from_numpy(inputs), 'target':torch.from_numpy(targets), 'encoding':torch.from_numpy(encodings)} #pylint: disable=no-member


    def transform_target(self, targets):
        return self.target_transform(targets)

    def transform_input(self, inputs):
        return self.input_transform(inputs) 

    def __repr__(self):
        pass

class QCIRCDataSetMulti(Dataset):
    """ QCIRC data set on lmdb."""
    def __init__(self, lmdb_dir, key_base = 'sample', input_transform=None, target_transform=None,
                                        input_shape=(1,85,120), target_shape=(3,),
                                        debug=True):
        self.debug = debug
        self.lmdb_path = [os.path.join(lmdb_dir, lmdb_path) for lmdb_path in os.listdir(lmdb_dir)]
        self.db = [lmdb.open(lmdb_path, readahead=False, readonly=True, writemap=False, lock=False)
                            for lmdb_path in self.lmdb_path]

        ## TODO: Need to specify how many records are for headers at __init__: now 2
        self.db_records = [db.stat()['entries'] - 2 for db in self.db]
        self.db_idx_sum = np.cumsum(self.db_records)

        with self.db[0].begin(write=False) as txn:
            self.dtype = np.dtype(txn.get(b"data_dtype"))
        self.print_debug("read dtype %s from lmdb file %s" %(format(self.dtype),
                                                            self.lmdb_path))
        #TODO: add shapes to lmdb headers.
        #TODO: add dtypes to lmbd headers.
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.key_base = key_base
        self.input_transform = input_transform
        self.target_transform = target_transform

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def __len__(self):
        ## TODO: Need to specify how many records are for headers at __init__: now 2
        return sum(self.db_records)


    def __getitem__(self, idx):
        # outside_func(idx)
        # map record index to lmdb file index
        db_idx = np.argmax(idx < self.db_idx_sum)
        if db_idx > 0: 
            idx -= np.sum(self.db_records[:db_idx])
        assert idx >= 0 and idx < sum(self.db_records) , print(idx, sum(self.db_records))
        # fetch records
        # try:
        with self.db[db_idx].begin(write=False, buffers=True) as txn:
            key = bytes('%s_%i' %(self.key_base, idx), "ascii")
            self.print_debug('going into lmdb file %s, reading %d' % (self.lmdb_path[db_idx], idx )) 
            bytes_buff = txn.get(key)
            sample = np.frombuffer(bytes_buff, dtype=self.dtype)
        input_size = np.prod(np.array(self.input_shape))
        target_size = np.prod(np.array(self.target_shape))
        input = sample[:input_size].astype('float32')
        target = sample[-target_size:].astype('float64')
        self.print_debug('read input %d with size %d' %(idx, input.size))
        if self.input_transform is not None:
            input = self.transform_input(input)
        if self.target_transform is not None:
            target = self.transform_target(target)

        input = input.reshape(self.input_shape)
        target = target.reshape(self.target_shape)

        return {'input':torch.from_numpy(input), 'target':torch.from_numpy(target)}
        # except AttributeError:
            # print("key: %s in file: %s" %(key, self.lmdb_path[db_idx]))

    @staticmethod
    def transform_target(target):
        if target.dtype != 'float64':
            return target.astype('float64')

    @staticmethod
    def transform_input(input):
        if input.dtype != 'float32':
            return input.astype('float32')

    def __repr__(self):
        pass

class QCIRCDataSetNumpy(Dataset):
    def __init__(self, file_path, input_transform=None, target_transform=None):
        self.data = np.load(file_path, mmap_mode='r')
        self.num_samples = self.data.shape[0]
        self.input_transform = input_transform
        self.target_transform = target_transform
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        inputs = self.data[idx,:,0]
        targets = self.data[idx,:,1]
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        if targets.dtype != np.float32:
            targets = targets.astype(np.float32)
        if self.input_transform:
            inputs = self.input_transform(inputs)
        if self.target_transform:
            targets = self.target_transform(targets)
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs) 
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets)
        return {'input':inputs, 'target':targets}
    
    def __repr__(self):
        pass

def set_io_affinity(mpi_rank, mpi_size, debug=True):
    """
    Set the affinity based on available cpus, mpi (local) rank, and mpi (local)
    size. Assumes mpirun binding is none.
    """
    if debug: 
        print("Initial Affinity %s" % os.sched_getaffinity(0))
    total_procs = len(os.sched_getaffinity(0))
    max_procs = total_procs // mpi_size
    new_affnty = range( mpi_rank * max_procs, (mpi_rank + 1) * max_procs)
    os.sched_setaffinity(0, new_affnty)
    if debug:
        print("New Affinity %s" % os.sched_getaffinity(0))
    return new_affnty

def benchmark_io(lmdb_path, mpi_rank, step=100, warmup=100, max_batches=1000,
                batch_size=512, shuffle=True, num_workers=20, pin_mem=True,
                gpu_copy=True, debug=False):
    """ Measure I/O performance of lmdb and multiple python processors during
        training.
    """
    QCIRCData = QCIRCDataSet(lmdb_path, debug=debug)
    data_loader = DataLoader(QCIRCData, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_mem)
    bandwidths=[]
    t = time()
    with torch.cuda.device(mpi_rank):
        for batch_num, batch in enumerate(data_loader):
            if gpu_copy:
                batch['input'].cuda(non_blocking=pin_mem)
            if batch_num % step == 0:
                print('loaded batch %d' % batch_num)
                print('input:', batch['input'].size(), 'target:', batch['target'].size())
                t_run = time() - t
                size = torch.prod(torch.tensor(batch['input'].size())).numpy() * \
                batch['input'].element_size() * step
                t = time()
                if batch_num > warmup:
                    bandwidths.append(size/1024e6/t_run)
                    print('Bandwidth: %2.3f (GB/s)' % (size/1024e6/t_run))
                t = time()
            if batch_num == max_batches:
                break
        bandwidths = np.array(bandwidths)
        print('Total Bandwidth: %2.2f +/- %2.2f (GB/s)' %(bandwidths.mean(), bandwidths.std()))
    return bandwidths.mean(), bandwidths.std()

def benchmark_io_multi(lmdb_path, mpi_rank, step=100, warmup=100, max_batches=1000,
                batch_size=512, shuffle=True, num_workers=20, pin_mem=True,
                gpu_copy=True, debug=False):
    """ Measure I/O performance of lmdb and multiple python processors during
        training.
    """
    QCIRCData = QCIRCDataSetMulti(lmdb_path, debug=debug)
    data_loader = DataLoader(QCIRCData, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_mem)
    bandwidths=[]
    t = time()
    with torch.cuda.device(mpi_rank):
        for batch_num, batch in enumerate(data_loader):
            if gpu_copy:
                batch['input'].cuda(non_blocking=pin_mem)
            if batch_num % step == 0:
                print('loaded batch %d' % batch_num)
                print('input:', batch['input'].size(), 'target:', batch['target'].size())
                t_run = time() - t
                size = torch.prod(torch.tensor(batch['input'].size())).numpy() * \
                batch['input'].element_size() * step
                t = time()
                if batch_num > warmup:
                    bandwidths.append(size/1024e6/t_run)
                    print('Bandwidth: %2.3f (GB/s)' % (size/1024e6/t_run))
                t = time()
            if batch_num == max_batches:
                break
        bandwidths = np.array(bandwidths)
        print('Total Bandwidth: %2.2f +/- %2.2f (GB/s)' %(bandwidths.mean(), bandwidths.std()))
    return bandwidths.mean(), bandwidths.std()
