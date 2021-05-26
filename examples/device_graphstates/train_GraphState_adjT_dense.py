import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from datetime import datetime

# qcdenoise imports
from qcdenoise import DenseModel, AdjTModel, AdjTAsymModel 
from qcdenoise.io_utils import QCIRCDataset
from qcdenoise.ml_utils import train, test

np.random.seed(1234)
# unitarynoise data
#data_path = '/data/MLQC/GraphState_nqbits9_UnitaryNoise_032920'
# noise_id = "Unitary"
# device noise data
data_path =  '/data/MLQC/GraphState_nqbits9_DeviceNoise_041420'
#data_path = '/data/MLQC/GraphState_nqbits5_DeviceNoise_041720'
#data_path='/data/MLQC/GraphState_nqbits5_DeviceNoise_042120/'
n_qbits = 9
noise_id = "Device"
traindata = QCIRCDataset('%s/train.lmdb' % data_path, pads=(0,32-n_qbits, 0, 32-n_qbits), debug=False)
testdata = QCIRCDataset('%s/dev.lmdb' % data_path, pads=(0, 32-n_qbits, 0, 32-n_qbits), debug=False)
print('Total # of samples in train set: {}, test set:{}'.format(len(traindata), len(testdata)))
batch_size=64
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
if not os.path.exists(data_path):
    os.mkdir(data_path)


# #### Plot a sample
idx = 0
inputs, targets, encodings =  traindata[idx]['input'], traindata[idx]['target'], traindata[idx]['encoding']
inputs_dim = inputs.shape[0]
targets_dim = targets.shape[0]
encodings_dim = encodings.shape
print("encodings shape", encodings_dim)


p_dropout=0.1
wd = 1e-4
inputs_dim = inputs.shape[0]
targets_dim = targets.shape[0]
encodings_dim = encodings.shape
net = AdjTAsymModel(n_qbits, inputs_dim=inputs_dim, targets_dim=targets_dim, encodings_dim=encodings_dim, 
                                  combine_mode='Multiply', asym_mode='dense', p_dropout=p_dropout)
print('AdjNet (dense Units):\n', net)


# ### Train

# #### loss functions


def mse(outputs, targets):
    MSE = torch.nn.MSELoss(reduction='sum')
    outputs = F.softmax(outputs, dim=1)
    return MSE(outputs, targets)

def kl(outputs, targets):
    KL = torch.nn.KLDivLoss(reduction='sum')
    outputs = F.log_softmax(outputs, dim=1)
    return KL(outputs, targets)


# #### Optimizer/Learning policy


def exp_scheduler(net, ilr=1e-3, lr_decay=0.9, weight_decay=1e-5):
    optimizer = optim.Adam(net.parameters(), lr=ilr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    return scheduler, optimizer

def step_scheduler(net, ilr=1e-3, lr_decay=0.1, weight_decay=1e-5):
    optimizer = optim.Adam(net.parameters(), lr=ilr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=lr_decay)
    return scheduler, optimizer

timestamp= datetime.now().strftime('%m%d%y')
model_path = os.path.join(data_path,'dense_model_dp{}_wd{}_{}.pt'.format(p_dropout, wd, timestamp))
loss_func = kl
# scheduler, optimizer = exp_scheduler(net, ilr=1e-3, lr_decay=0.95, weight_decay=wd)
scheduler, optimizer = step_scheduler(net, ilr=1e-3, lr_decay=0.5, weight_decay=wd)
test_func_args = (net, testloader, loss_func)
res_logs = train(net, trainloader, loss_func, scheduler, optimizer, save_epoch=1, lr=1e-3, 
      dev_num=0, step_log= 500, num_epochs=50, test_epoch=1, test_func_args=test_func_args, path=model_path)


# In[ ]:


logs = res_logs
fig, ax = plt.subplots()
ax.plot(logs["step"], logs["loss"], label="Train", marker='o', linestyle='dashed')
ax.plot(logs["test_step"], logs["test_loss"], label="Dev", marker='^', linestyle='dashed')
ax.set_xlabel("Iterations", fontsize=14)
ax.set_ylabel("Loss (KL Divergence)", fontsize=14)
ax.legend()
fig.savefig(os.path.join(data_path,"trainStats_dp{}_wd{}_dense.png".format(p_dropout, wd)), dpi=300)

log_fp = os.path.join(data_path, "logs_dp{}_wd{}_dense.json".format(p_dropout, wd))
with open(log_fp, mode='w') as fp:
    json.dump(logs, fp)

# #### Load Trained Model from checkpoint

# In[67]:


net = AdjTAsymModel(inputs_dim=inputs_dim, targets_dim=targets_dim, encodings_dim=encodings_dim, 
                        combine_mode='Multiply', asym_mode='dense')
net.load_state_dict(torch.load(model_path))


testdata = QCIRCDataset('%s/test.lmdb' % data_path, debug=False)
print('Total # of samples in train set: {}, test set:{}'.format(len(traindata), len(testdata)))
trainloader = DataLoader(traindata, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)

kl_test = test(net, testloader, kl)


# In[87]:


idx = np.random.randint(0, len(testdata)-1)
print('sample=%d'%idx)
inputs, targets, encodings = testdata[idx]['input'], testdata[idx]['target'], testdata[idx]['encoding']
with torch.no_grad():
    inputs = torch.unsqueeze(inputs,0)
    encodings = torch.unsqueeze(encodings, 0)
    net = net.to('cpu')
    net.eval()
    outputs_res = net(inputs, encodings)
    outputs_res = F.softmax(outputs_res)
fig,axes = plt.subplots(1,2,figsize=(14,6), sharex=True, sharey=True)
# ax.plot(np.squeeze(outputs.numpy()), label='Ouput', marker='s')
# ax.plot(np.squeeze(outputs_dense.numpy()), label='Ouput- dense', marker='o')
axes[0].bar(np.arange(inputs_dim), np.squeeze(outputs_res.numpy()), width=4, label='ML Ouput- dense')
axes[0].bar(np.arange(inputs_dim), np.squeeze(targets.numpy()), width=4, label='Target- Ideal Circuit', color='w', edgecolor='r', alpha=0.35)
axes[1].bar(np.arange(inputs_dim), np.squeeze(outputs_res.numpy()), width=4, label='ML Ouput- dense')
axes[1].bar(np.arange(inputs_dim), np.squeeze(inputs.numpy()), width=4, label='Input- %s Noise' % noise_id, color='w', edgecolor='k', alpha=0.35)
# axes[0].set_title('Test Sample #%d' %idx, fontsize=12)
axes[0].set_title('Test Sample #%d AdjTAsym Model\nTrained on 9-qubit Graph States.\nTested: on different Graph States'%idx, fontsize=12)
axes[1].set_title('Test Sample #%d AdjTAsym Model\nKL per sample = %2.2f' %(idx, kl_test/batch_size) ,fontsize=12)
axes[0].set_xlabel("Computational Basis ($2^N$ Possible Outcomes)", fontsize=14)
axes[0].set_ylabel("Probability", fontsize=14)
axes[0].legend()
axes[1].legend()
fig.savefig(os.path.join(data_path, "Results_dp{}_wd{}_dense.png".format(p_dropout, wd)), dpi=300)


