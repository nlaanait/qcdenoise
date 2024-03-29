from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .config import get_module_logger
#pylint: disable=no-member

# module logger
logger = get_module_logger(__name__)


class DenseModel(nn.Module):

    def __init__(self, inputs_dim=None, targets_dim=None):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(inputs_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 512)
        self.fc7 = nn.Linear(512, targets_dim)
        self.fc8 = nn.Linear(targets_dim, targets_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class AdjTModel(nn.Module):
    def __init__(self, inputs_dim=None, targets_dim=None,
                 encodings_dim=None, combine_mode='Add'):
        super(AdjTModel, self).__init__()
        self.encodings_dim = encodings_dim
        self.combine_mode = combine_mode
        self.targets_dim = targets_dim
        self.fc1 = nn.Linear(inputs_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, targets_dim)
        self.fc8 = nn.Linear(targets_dim, targets_dim)
        self.conv1 = nn.Conv2d(
            self.encodings_dim[0],
            32,
            3,
            padding=3 // 2,
            bias=False)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=3 // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=3 // 2, bias=False)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(64, self.targets_dim // (self.encodings_dim[1] * self.encodings_dim[2]), 3, padding=3 // 2,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

    def forward(self, prob, adjT):
        prob = F.relu(self.fc1(prob))
        prob = F.relu(self.fc2(prob))
        prob = F.relu(self.fc3(prob))
        prob = F.relu(self.fc4(prob))
        prob = F.relu(self.fc5(prob))
        prob = F.relu(self.fc6(prob))
        prob = self.fc7(prob)

        adjT = F.relu(self.bn1(self.conv1(adjT)))
        adjT = F.relu(self.bn2(self.conv2(adjT)))
        adjT = F.relu(self.bn3(self.conv3(adjT)))
        adjT = F.relu(self.bn4(self.conv4(adjT)))

        # combine output of both branches
        x = self.combine(prob, adjT)
        x = F.relu(self.fc8(x))
        return x

    def combine(self, x, y):
        if self.combine_mode == 'Add':
            y = y.view(x.shape)
            return x + y
        elif self.combine_mode == 'Multiply':
            y = y.view(-1, self.targets_dim)
            return x * y


class AdjTAsymModel(nn.Module):
    def __init__(self, n_qubits, inputs_dim=None, targets_dim=None, encodings_dim=None,
                 combine_mode='Add', asym_mode='dense', out_c=32, p_dropout=0.25):
        assert asym_mode in [
            'residual', 'dense'], 'asym_mode requested is not implemented'
        super(AdjTAsymModel, self).__init__()
        self.encodings_dim = encodings_dim
        self.combine_mode = combine_mode
        self.targets_dim = targets_dim
        self.asym_mode = asym_mode
        # layers for prob vector
        self.fc1 = nn.Linear(inputs_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, targets_dim)
        # layers for adjacency tensor
        adj_c = n_qubits
        kern = [3, 3]
        pad = [kern[0] // 2, kern[1] // 2]
        kern_v = [max(adj_c + adj_c % 2 - 1, 3), 3]
        kern_h = kern_v[::-1]
        pad_v = [kern_v[0] // 2, kern_v[1] // 2]
        pad_h = pad_v[::-1]
        stride_v = [1, 1]
        stride_h = stride_v[::-1]
        self.conv0 = nn.Conv2d(
            self.encodings_dim[0],
            out_c,
            kern,
            padding=pad,
            bias=True)
        in_c = self.conv0.out_channels
        self.conv1_v = nn.Conv2d(
            in_c,
            out_c,
            kern_v,
            padding=pad_v,
            stride=stride_v,
            bias=False)
        self.conv1_h = nn.Conv2d(
            out_c,
            out_c,
            kern_h,
            padding=pad_h,
            stride=stride_h,
            bias=False)
        in_c = self.calc_in_c(in_c, self.conv1_h)
        self.conv1 = nn.Conv2d(
            in_c,
            out_c * 2,
            kern,
            padding=pad,
            bias=False)
        in_c = self.conv1.out_channels
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2_v = nn.Conv2d(
            self.conv1.out_channels,
            out_c * 2,
            kern_v,
            padding=pad_v,
            stride=stride_v,
            bias=False)
        self.conv2_h = nn.Conv2d(
            self.conv2_v.out_channels,
            out_c * 2,
            kern_h,
            padding=pad_h,
            stride=stride_h,
            bias=False)
        in_c = self.calc_in_c(in_c, self.conv2_h)
        self.conv2 = nn.Conv2d(
            in_c,
            out_c * 2,
            kern,
            padding=pad,
            bias=False)
        in_c = self.conv2.out_channels
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3_v = nn.Conv2d(
            self.conv2.out_channels,
            out_c * 2,
            kern_v,
            padding=pad_v,
            stride=stride_v,
            bias=False)
        self.conv3_h = nn.Conv2d(
            self.conv3_v.out_channels,
            out_c * 2,
            kern_h,
            padding=pad_h,
            stride=stride_h,
            bias=False)
        in_c = self.calc_in_c(in_c, self.conv3_h)
        self.conv3 = nn.Conv2d(
            in_c,
            out_c * 2,
            kern,
            padding=pad,
            bias=False)
        in_c = self.conv3.out_channels
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4_v = nn.Conv2d(
            self.conv3.out_channels,
            out_c * 2,
            kern_v,
            padding=pad_v,
            stride=stride_v,
            bias=False)
        self.conv4_h = nn.Conv2d(
            self.conv4_v.out_channels,
            out_c * 2,
            kern_h,
            padding=pad_h,
            stride=stride_h,
            bias=False)
        in_c = self.calc_in_c(in_c, self.conv4_h)
        self.conv4 = nn.Conv2d(in_c, max(self.targets_dim // (self.encodings_dim[1] * self.encodings_dim[2]), 1), kern, padding=pad,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)
        adjT_shape, prob_shape = self.test_forward(torch.zeros(
            [1] + [inputs_dim]), torch.zeros([1] + list(self.encodings_dim)))
        self.adjT_flat_size = adjT_shape[1] * \
            adjT_shape[2] * adjT_shape[3]
        self.fcAdjT = nn.Linear(self.adjT_flat_size, targets_dim)
        self.fcFinal = nn.Linear(targets_dim, targets_dim)
        if p_dropout < 0 or p_dropout > 1:
            warn(
                "invalid p_dropout value was passed. Defaulting to p_dropout=0")
            p_dropout = 0
        self.drop2d = nn.Dropout2d(p=p_dropout)
        self.drop1d = nn.Dropout(p=p_dropout)

    def calc_in_c(self, in_c, prev_conv):
        if self.asym_mode == 'residual':
            in_c = prev_conv.out_channels
        elif self.asym_mode == 'dense':
            in_c += prev_conv.out_channels
        return in_c

    def test_forward(self, prob, adjT):
        # forward on prob vec branch
        prob = F.relu(self.fc1(prob))
        prob = F.relu(self.fc2(prob))
        prob = F.relu(self.fc3(prob))
        prob = F.relu(self.fc4(prob))
        prob = F.relu(self.fc5(prob))
        prob = F.relu(self.fc6(prob))
        prob = self.fc7(prob)

        # forward on adjacency tensor branch
        if self.asym_mode == 'residual':
            asym_block = self.residual_block
        elif self.asym_mode == 'dense':
            asym_block = self.dense_block

        adjT = self.conv0(adjT)
        adjT = asym_block(
            adjT,
            self.bn1,
            self.conv1,
            self.conv1_h,
            self.conv1_v)
        adjT = asym_block(
            adjT,
            self.bn2,
            self.conv2,
            self.conv2_h,
            self.conv2_v)
        adjT = asym_block(
            adjT,
            self.bn3,
            self.conv3,
            self.conv3_h,
            self.conv3_v)
        adjT = asym_block(
            adjT,
            self.bn4,
            self.conv4,
            self.conv4_h,
            self.conv4_v)
        return adjT.shape, prob.shape

    def forward(self, prob, adjT):
        # forward on prob vec branch
        prob = self.drop1d(F.relu(self.fc1(prob)))
        prob = self.drop1d(F.relu(self.fc2(prob)))
        prob = self.drop1d(F.relu(self.fc3(prob)))
        prob = self.drop1d(F.relu(self.fc4(prob)))
        prob = self.drop1d(F.relu(self.fc5(prob)))
        prob = self.drop1d(F.relu(self.fc6(prob)))
        prob = self.fc7(prob)

        # forward on adjacency tensor branch
        if self.asym_mode == 'residual':
            asym_block = self.residual_block
        elif self.asym_mode == 'dense':
            asym_block = self.dense_block

        adjT = self.conv0(adjT)
        adjT = self.drop2d(
            asym_block(
                adjT,
                self.bn1,
                self.conv1,
                self.conv1_h,
                self.conv1_v))
        adjT = self.drop2d(
            asym_block(
                adjT,
                self.bn2,
                self.conv2,
                self.conv2_h,
                self.conv2_v))
        adjT = self.drop2d(
            asym_block(
                adjT,
                self.bn3,
                self.conv3,
                self.conv3_h,
                self.conv3_v))
        adjT = self.drop2d(
            asym_block(
                adjT,
                self.bn4,
                self.conv4,
                self.conv4_h,
                self.conv4_v))

        # combine output of both branches
        x = self.combine(prob, adjT)
        x = F.relu(self.fcFinal(x))
        return x

    def dense_block_forked(self, x, bn, conv, conv_h, conv_v):
        out = x
        x_v = F.relu(conv_v(x))
        x_h = F.relu(conv_h(x))
        x = x_v * x_h
        x = torch.cat([out, x], dim=1)
        x = F.relu(bn(conv(x)))
        return x

    def dense_block(self, x, bn, conv, conv_h, conv_v):
        out = x
        x = F.relu(conv_v(x))
        x = F.relu(conv_h(x))
        x = torch.cat([out, x], dim=1)
        x = F.relu(bn(conv(x)))
        return x

    def residual_block(self, x, bn, conv, conv_h, conv_v):
        residual = x
        x = F.relu(conv_v(x))
        x = F.relu(conv_h(x))
        x += residual
        x = F.relu(bn(conv(x)))
        return x

    def combine(self, x, y):
        if self.combine_mode == 'Add':
            y = y.view(-1, self.adjT_flat_size)
            y = self.fcAdjT(y)
            return x + y
        elif self.combine_mode == 'Multiply':
            y = y.view(-1, self.adjT_flat_size)
            y = self.fcAdjT(y)
            return x * y

    def get_stats(self):
        n_params = 0
        for param_tensor in self.state_dict():
            tensor_size = self.state_dict()[param_tensor].size()
            n_params += np.prod(tensor_size[:])
            print(param_tensor, "\t", tensor_size)
        print("Total # of weights = {}".format(int(n_params)))


if __name__ == "__main__":
    inputs_dim = 256
    targets_dim = 256
    encodings_dim = [8, 8, 8]
    net = AdjTAsymModel(
        8,
        inputs_dim=inputs_dim,
        targets_dim=targets_dim,
        encodings_dim=encodings_dim)
    print(net)
