from time import time
import os
import subprocess
import shlex
import random
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TT

def pool_shuffle_split(files_dir, file_expr, split=0.8, delete=True):
    files = os.listdir(files_dir)
    files = [file for file in files if file_expr in file]
    runs = np.concatenate([np.load(os.path.join(files_dir, file), mmap_mode='r') for file in files])
    np.random.shuffle(runs)
    part = int(runs.shape[0] * split)
    train = runs[:part]
    test = runs[part:]
    np.save(file_expr+'train.npy', train)
    print('wrote {} with shape {}'.format(file_expr+'train.npy', train.shape))
    np.save(file_expr+'test.npy', test)
    print('wrote {} with shape {}'.format(file_expr+'test.npy', test.shape))
    cond = os.path.exists(file_expr+'train.npy') and os.path.exists(file_expr+'test.npy')
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

def numpy_to_lmdb(lmdb_path, data, labels, lmdb_map_size=int(50e9)):
    env = lmdb.open(lmdb_path, map_size=lmdb_map_size, map_async=True, writemap=True, create=True)
    with env.begin(write=True) as txn:
        for (i, datum) , label in zip(enumerate(data), labels):
            key = bytes('input_%s'%format(i), "ascii")
            inputs_shape = datum.shape
            outputs_shape = label.shape
            inputs = datum.flatten().tostring()
            txn.put(key, inputs)
            key = bytes('output_%s'%format(i), "ascii")
            outputs = label.flatten().tostring()
            txn.put(key, outputs)
        env.sync()
        headers = { b"input_dtype": bytes(datum.dtype.str, "ascii"),
                    b"input_shape": np.array(inputs_shape).tostring(),
                    b"output_shape": np.array(outputs_shape).tostring(),
                    b"output_dtype": bytes(label.dtype.str, "ascii"),
                    b"output_name": bytes('output_', "ascii"),
                    b"input_name": bytes('input_', "ascii")}
        for key, val in headers.items():
            txn.put(key, val)
        txn.put(b"header_entries", bytes(len(list(headers.items()))))
        env.sync()

class QCIRCTransform:

    """Apply AffineTransform --> RandomCrop --> Uniform Noise"""

    def __init__(self, crop_shape=(64,94), angle=(-5,5), scale=(1,1.1), shear=(-3,3), noise=(0,0.1), translate=(0,1)):
        """__init__ [summary]
        
        Args:
            crop_shape (tuple, optional): [description]. Defaults to (64,94).
            angle (tuple, optional): [description]. Defaults to (-5,5).
            scale (tuple, optional): [description]. Defaults to (1, 1.2).
            translate (tuple, optional): [description]. Defaults to (0,1).
            shear (tuple, optional): [description]. Defaults to (-5,5).
            noise (tuple, optional): [description]. Defaults to (0,0.4).
        """
        self.crop_shape = crop_shape
        self.angle = angle
        self.scale = scale
        self.translate = translate
        self.shear = shear
        self.noise = noise
        return 
    
    def _get_crops_params(self, shape):
        """_get_crops_params [summary]
        
        Args:
            shape ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        h=self.crop_shape[0]
        w=self.crop_shape[1]
        img_h, img_w = shape
        i= random.uniform(0, img_h-h)
        j= random.uniform(0, img_w-w)
        return (i, j, h, w)
    
    def _get_affine_params(self):
        """_get_affine_params [summary]
        
        Returns:
            [type]: [description]
        """
        angle = random.uniform(self.angle[0], self.angle[1])
        translate = random.uniform(self.translate[0], self.translate[1])
        translate = (translate,)*2
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(self.shear[0], self.shear[1])
        return (angle, translate, scale, shear)
        
    def _get_noise_tensor(self, shape):
        """_get_noise_tensor [summary]
        
        Args:
            shape ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        noise_tens = torch.randint(2, shape, dtype=torch.float32)
        noise_tens *= random.uniform(self.noise[0], self.noise[1])
        return noise_tens
    
    def __call__(self, x):
        """__call__ [summary]
        
        Args:
            x ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        if len(x.shape) < 3:
            x = np.expand_dims(x, axis=-1)
        x = x.astype(np.float32)
        # get random transform params
        crops_params = self._get_crops_params(x.shape[:-1])
        affine_params = self._get_affine_params()
        # convert to PIL
        toPIL = TT.ToPILImage()
        x = toPIL(x)
        # apply transforms
        x = TT.functional.affine(x, *affine_params,resample=3)
        x = TT.functional.crop(x, *crops_params)
        # convert to tensor and add noise
        toTens = TT.ToTensor()
        x = toTens(x)
        noise = self._get_noise_tensor(x.shape)
        x +=  noise
        x[x > 1.001] = 1
        x[x < 0.001] = 0
        return x

class QCIRCDataSet(Dataset):
    """ QCIRC data set on lmdb."""
    def __init__(self, lmdb_path, key_base = 'sample', input_transform=QCIRCTransform(), target_transform=None,
                                        debug=True):
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
        self.num_samples = (self.env.stat()['entries'] - 6)//2 ## TODO: remove hard-coded # of headers by storing #samples key, val
        self.first_record = 0
        self.records = np.arange(self.first_record, self.num_samples)
        with self.env.begin(write=False) as txn:
            input_shape = np.frombuffer(txn.get(b"input_shape"), dtype='int64')
            output_shape = np.frombuffer(txn.get(b"output_shape"), dtype='int64')
            input_dtype = np.dtype(txn.get(b"input_dtype").decode("ascii"))
            output_dtype = np.dtype(txn.get(b"output_dtype").decode("ascii"))
            output_name = txn.get(b"output_name").decode("ascii")
            input_name = txn.get(b"input_name").decode("ascii")
        self.data_specs={'input_shape': list(input_shape), 'target_shape': list(output_shape), 
            'target_dtype':output_dtype, 'input_dtype': input_dtype, 'target_key':output_name, 'input_key': input_name}
        self.input_keys = [bytes(self.data_specs['input_key']+str(idx), "ascii") for idx in self.records]
        self.target_keys = [bytes(self.data_specs['target_key']+str(idx), "ascii") for idx in self.records]
        self.print_debug("Opened lmdb file %s, with %d samples" %(self.lmdb_path, self.num_samples))
        self.input_transform = input_transform
        self.target_transform = target_transform

    def print_debug(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def __len__(self):
        ## TODO: Need to specify how many records are for headers
        return self.num_samples

    def __getitem__(self, idx):
        # outside_func(idx)
        input_key = self.input_keys[idx]
        target_key = self.target_keys[idx]
        with self.env.begin(write=False, buffers=True) as txn:
            input_bytes = txn.get(input_key)
            target_bytes = txn.get(target_key)
        inputs = np.frombuffer(input_bytes, dtype=self.data_specs['input_dtype'])
        inputs = inputs.reshape(self.data_specs['input_shape'])
        targets = np.frombuffer(target_bytes, dtype=self.data_specs['target_dtype'])
        targets = targets.reshape(self.data_specs['target_shape'])
        self.print_debug('read (inputs, targets) # {idx} with shape ({in_shape}, {trg_shape})'.format(idx=idx, in_shape=inputs.shape, trg_shape=targets.shape))
        if self.input_transform is not None:
            inputs = self.transform_input(inputs)
        if self.target_transform is not None:
            targets = self.transform_target(targets)
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs) 
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets)
        return {'input':inputs, 'target':targets}

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
