import torch
import torch.optim as optim


def train(net, dataloader, loss_func, dev_num=0, lr=1e-4, weight_decay=1e-4, num_epochs=10, batch_log=500, test_epoch=2, test_func_args=None):
    running_loss = 0.0 
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    dev_name = "cuda:%d" %dev_num
    device = torch.device(dev_name if torch.cuda.is_available() else "cpu") #pylint: disable=no-member
    net.train()
    for epoch in range(num_epochs):
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
                net.eval()
                test_func, test_args = test_func_args
                print('Test Data:')
                test_func(*test_args)
                net.train()
            else:
                print('Skipping Evaluation: test_func_args was not provided')

    return net

def test(net, dataloader, loss_func, dev_num=0):
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
    return test_loss