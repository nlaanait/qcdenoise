import torch
import os
import torch.optim as optim
from qcdenoise.ml_models import *

def train(*args, **kwargs):
    net = args[0]
    val = None
    if isinstance(net, AdjTModel) or isinstance(net, AdjTAsymModel):
        val = train_adjT(*args, **kwargs)
    elif isinstance(net, DenseModel):
        val = train_Dense(*args, **kwargs)
    return val


def test(*args, **kwargs):
    net = args[0]
    if isinstance(net, AdjTModel) or isinstance(net, AdjTAsymModel):
        test_AdjT(*args, **kwargs)
    elif isinstance(net, DenseModel):
        test_Dense(*args, **kwargs)


def train_Dense(net, dataloader, loss_func, dev_num=0, lr=1e-4, weight_decay=1e-4, num_epochs=10, batch_log=500,
                test_epoch=2, test_func_args=None):
    running_loss = 0.0 
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    dev_name = "cuda:%d" %dev_num
    device = torch.device(dev_name if torch.cuda.is_available() else "cpu") #pylint: disable=no-member
    net.train()
    opt_state = None
    for epoch in range(num_epochs):
        if opt_state:
            optimizer.load_state_dict(opt_state)
        for batch_num, batch in enumerate(dataloader):
            # data
            inputs, targets = batch['input'], batch['target']
            inputs = inputs.to(device)
            targets = targets.to(device)
            net = net.to(device)
            # forward
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            # train
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print stats
            running_loss += loss.item()
            if batch_num % batch_log == batch_log - 1:    
                print('Epoch={}, Batch={:5d}, Loss= {:.3f}'.format(epoch + 1, batch_num + 1, running_loss / batch_log))
                running_loss = 0.0
        if epoch % test_epoch == test_epoch - 1:
            if test_func_args:
                opt_state = optimizer.state_dict()
                net.eval()
                print('Test Data:')
                test_Dense(*test_func_args)
                net.train()
            else:
                print('Skipping Evaluation: test_func_args was not provided')

    # return net

def test_Dense(net, dataloader, loss_func, dev_num=0):
    net.eval()
    dev_num = 0
    running_loss = 0
    dev_name = "cuda:%d" %dev_num
    device = torch.device(dev_name if torch.cuda.is_available() else "cpu") #pylint: disable=no-member
    with torch.no_grad():
        for batch_num, batch in enumerate(dataloader):
            # data
            inputs, targets = batch['input'], batch['target']
            inputs = inputs.to(device)
            targets = targets.to(device)
            net = net.to(device)
            # forward
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            # print stats
            running_loss += loss.item()
        test_loss = running_loss / (batch_num + 1)
        print('Batches={}, Average Loss= {:.3f}'.format(batch_num + 1, test_loss)) 
    # return test_loss

def test_AdjT(net, dataloader, loss_func, dev_num=0):
    net.eval()
    dev_num = 0
    running_loss = 0
    dev_name = "cuda:%d" %dev_num
    device = torch.device(dev_name if torch.cuda.is_available() else "cpu") #pylint: disable=no-member
    with torch.no_grad():
        for batch_num, batch in enumerate(dataloader):
            # data
            inputs, targets, encodings = batch['input'], batch['target'], batch['encoding']
            inputs = inputs.to(device)
            targets = targets.to(device)
            encodings = encodings.to(device)
            net = net.to(device)
            # forward
            outputs = net(inputs, encodings)
            loss = loss_func(outputs, targets)
            # print stats
            running_loss += loss.item()
        test_loss = running_loss / (batch_num + 1)
        print('Batches={}, Average Loss= {:.3f}'.format(batch_num + 1, test_loss)) 
    return test_loss

def train_adjT(net, dataloader, loss_func, scheduler, optimizer, dev_num=0, lr=1e-4, weight_decay=1e-4, 
               num_epochs=10, step_log=500, test_epoch=2, save_epoch=5, path='data/model.pt', test_func_args=None):
    running_loss = 0.0 
    dev_name = "cuda:%d" %dev_num
    device = torch.device(dev_name if torch.cuda.is_available() else "cpu") #pylint: disable=no-member
    logs = {'lr':[], 'epoch':[], 'step':[], 'loss':[], 'test_loss':[], 'test_step':[]}
    opt_state = None
    step = 0
    if test_func_args:
        opt_state = optimizer.state_dict()
        epoch_test = test_epoch
        net.eval()
        print('Test Data:')
        test_loss = test_AdjT(*test_func_args)
        logs['test_loss'].append(test_loss)
        logs['test_step'].append(step)
    net.train()
    for epoch in range(num_epochs):
        scheduler.step()
        if opt_state:
            optimizer.load_state_dict(opt_state)
        for batch_num, batch in enumerate(dataloader):
            step += 1
            # data
            inputs, targets, encodings = batch['input'], batch['target'], batch['encoding']
            inputs = inputs.to(device)
            targets = targets.to(device)
            encodings = encodings.to(device)
            net = net.to(device)
            # forward
            optimizer.zero_grad()
            outputs = net(inputs, encodings)
            loss = loss_func(outputs, targets)
            # train
            loss.backward()
            optimizer.step()
            # print stats
            running_loss += loss.item()
            if step % step_log == 0 :
                logs['lr'].append(scheduler.get_lr()[0])
                logs['epoch'].append(epoch+1)
                logs['step'].append(step)
                logs['loss'].append(running_loss/step_log)    
                print('Epoch={}, Lr= {:5f}, Step={:5d}, Loss={:.3f}'.format(logs['epoch'][-1], logs['lr'][-1], 
                      logs['step'][-1], logs['loss'][-1]))
                running_loss = 0.0
        if epoch % test_epoch == test_epoch - 1:
            if test_func_args:
                opt_state = optimizer.state_dict()
                net.eval()
                print('Test Data:')
                test_loss = test_AdjT(*test_func_args)
                epoch_test += test_epoch
                logs['test_loss'].append(test_loss)
                logs['test_step'].append(step)
                net.train()
            else:
                print('Skipping Evaluation: test_func_args was not provided')
        if epoch % save_epoch == save_epoch - 1:
            torch.save(net.state_dict(), path)
    return logs
