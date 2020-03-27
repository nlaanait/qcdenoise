import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#pylint: disable=no-member

def build_gsn(input_dim, hidden_nodes=1024, hidden_nodes_stochastic=16, walkback=6, noise_rate=0.2):
    x_0 = tf.placeholder(tf.float32, [None, input_dim])
    W1 = get_shared_weights(input_dim, hidden_nodes, np.sqrt(1 / (hidden_nodes + input_dim)))
    W2 = get_shared_weights(hidden_nodes, hidden_nodes, np.sqrt(6. / (2 * hidden_nodes)))
    b0 = get_shared_bias(input_dim)
    b1 = get_shared_bias(hidden_nodes)
    b2  = get_shared_bias(hidden_nodes)
    x_0_next = x_0

    x_chain = []
    h_2 = tf.zeros([1, hidden_nodes])
    # build denoiser chain
    for _ in range(walkback):
        # Add noise
        x_corrupt = salt_and_pepper(x_0_next, noise_rate)
        # Activate
        h_1 = tf.tanh(tf.matmul(x_corrupt, W1) + tf.matmul(h_2, tf.transpose(W2)) + b1)
        # Activate
        h_2 = add_gaussian_noise(tf.tanh(add_gaussian_noise(tf.matmul(h_1, W2) + b2, hidden_nodes_stochastic)), hidden_nodes_stochastic)
        # Activate
        x_1 = tf.sigmoid(tf.matmul(h_1, tf.transpose(W1)) + b0)
        # Build the reconstruction chain
        x_chain.append(x_1)
        # Input sampling
        x_0_next = binomial_draw_vec(x_1)
    
    probs = [tf.cross_entropy(x_0, x_1) for x_1 in x_chain]
    cross_entropy = tf.add_n(probs)
    return cross_entropy
    
def binomial_draw(shape=[1], p=0.5, dtype='float32'):
      return tf.select(tf.less(tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32'), tf.fill(shape, p)), tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))

def binomial_draw_vec(p_vec, dtype='float32'):
  shape = tf.shape(p_vec)
  return tf.select(tf.less(tf.random_uniform(shape=shape, minval=0, maxval=1, dtype='float32'), p_vec), tf.ones(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))

def salt_and_pepper(X, rate=0.3):
  a = binomial_draw(shape=tf.shape(X), p=1-rate)
  b = binomial_draw(shape=tf.shape(X), p=0.5)
  z = tf.zeros(tf.shape(X), dtype='float32')
  c = tf.select(tf.equal(a, z), b, z)
  return tf.add(tf.mul(X, a), c)

def add_gaussian_noise(X, sigma):
  noise = tf.random_normal(tf.shape(X), stddev=sigma, dtype=tf.float32)
  return tf.add(X, noise)

# Xavier Initializers
def get_shared_weights(n_in, n_out, interval):
    val = np.random.uniform(-interval, interval, size=(n_in, n_out))
    val = val.astype(np.float32)
    return tf.Variable(val)

def get_shared_bias(n, offset = 0):
    val = np.zeros(n) - offset
    val = val.astype(np.float32)
    return tf.Variable(val)

    
class DenseModel(nn.Module):

    def __init__(self, inputs_dim=None, targets_dim=None):
        super(DenseModel, self).__init__()
        self.fc1  = nn.Linear(inputs_dim, 512)
        self.fc2 =  nn.Linear(512, 512)
        self.fc3 =  nn.Linear(512, 1024)
        self.fc4 =  nn.Linear(1024, 1024)
        self.fc5 =  nn.Linear(1024, 512)
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
  def __init__(self, inputs_dim=None, targets_dim=None, encodings_dim=None, combine_mode='Add'):
    super(AdjTModel, self).__init__()
    self.encodings_dim = encodings_dim
    self.combine_mode = combine_mode
    self.targets_dim = targets_dim
    self.fc1  = nn.Linear(inputs_dim, 256)
    self.fc2 =  nn.Linear(256, 512)
    self.fc3 =  nn.Linear(512, 512)
    self.fc4 =  nn.Linear(512, 512)
    self.fc5 =  nn.Linear(512, 256)
    self.fc6 = nn.Linear(256, 256)  
    self.fc7 = nn.Linear(256, targets_dim)
    self.fc8 = nn.Linear(targets_dim, targets_dim)
    self.conv1 = nn.Conv2d(self.encodings_dim[0], 32, 3, padding=3//2, bias=False)
    self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=3//2, bias=False)
    self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
    self.conv3 = nn.Conv2d(64, 64, 3, padding=3//2, bias=False)
    self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
    self.conv4 = nn.Conv2d(64, self.targets_dim//(self.encodings_dim[1]*self.encodings_dim[2]), 3, padding=3//2, bias=False)
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
  def __init__(self, inputs_dim=None, targets_dim=None, encodings_dim=None, 
                     combine_mode='Add', asym_mode='dense'):
    assert asym_mode in ['residual', 'dense'], 'asym_mode requested is not implemented'
    super(AdjTAsymModel, self).__init__()
    self.encodings_dim = encodings_dim
    self.combine_mode = combine_mode
    self.targets_dim = targets_dim
    self.asym_mode = asym_mode
    # layers for prob vector
    self.fc1  = nn.Linear(inputs_dim, 256)
    self.fc2 =  nn.Linear(256, 512)
    self.fc3 =  nn.Linear(512, 512)
    self.fc4 =  nn.Linear(512, 512)
    self.fc5 =  nn.Linear(512, 256)
    self.fc6 = nn.Linear(256, 256)  
    self.fc7 = nn.Linear(256, targets_dim)
    # self.fc8 = nn.Linear(targets_dim, targets_dim)
    # layers for adjacency tensor
    adj_c = self.encodings_dim[2]
    kern_v = [adj_c + adj_c%2 -1, 1]
    kern_h = kern_v[::-1]
    pad_v = [kern_v[0]//2, kern_v[1]//2]
    pad_h = pad_v[::-1]
    stride_v = [1,1]
    stride_h = stride_v[::-1]
    out_c = 32
    self.conv0 = nn.Conv2d(self.encodings_dim[0], out_c, 3, padding=3//2, bias=True)
    in_c = self.conv0.out_channels
    self.conv1_v = nn.Conv2d(in_c, out_c, kern_v, padding=pad_v, stride=stride_v, bias=False)
    self.conv1_h = nn.Conv2d(out_c, out_c, kern_h, padding=pad_h, stride=stride_h, bias=False)
    in_c = self.calc_in_c(in_c, self.conv1_h)
    self.conv1 = nn.Conv2d(in_c, out_c * 2, 3, padding=3//2, bias=False)
    in_c = self.conv1.out_channels
    self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
    self.conv2_v = nn.Conv2d(self.conv1.out_channels, out_c*2, kern_v, padding=pad_v, stride=stride_v, bias=False)
    self.conv2_h = nn.Conv2d(self.conv2_v.out_channels, out_c*2, kern_h, padding=pad_h, stride=stride_h, bias=False)
    in_c = self.calc_in_c(in_c, self.conv2_h) 
    self.conv2 = nn.Conv2d(in_c, out_c * 2, 3, padding=3//2, bias=False)
    in_c = self.conv2.out_channels
    self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
    self.conv3_v = nn.Conv2d(self.conv2.out_channels, out_c*2, kern_v, padding=pad_v, stride=stride_v, bias=False)
    self.conv3_h = nn.Conv2d(self.conv3_v.out_channels, out_c*2, kern_h, padding=pad_h, stride=stride_h, bias=False)
    in_c = self.calc_in_c(in_c, self.conv3_h)
    self.conv3 = nn.Conv2d(in_c, out_c * 2, 3, padding=3//2, bias=False)
    in_c = self.conv3.out_channels
    self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
    self.conv4_v = nn.Conv2d(self.conv3.out_channels, out_c*2, kern_v, padding=pad_v, stride=stride_v, bias=False)
    self.conv4_h = nn.Conv2d(self.conv4_v.out_channels, out_c*2, kern_h, padding=pad_h, stride=stride_h, bias=False)
    in_c = self.calc_in_c(in_c, self.conv4_h)
    self.conv4 = nn.Conv2d(in_c, self.targets_dim//(self.encodings_dim[1]*self.encodings_dim[2]), 3, padding=3//2, bias=False)
    self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)
    adjT_shape, prob_shape = self.test_forward(torch.zeros([1]+[inputs_dim]), torch.zeros([1]+list(self.encodings_dim)))
    self.adjT_flat_size = adjT_shape[1]*adjT_shape[2]*adjT_shape[3]
    self.fcAdjT = nn.Linear(self.adjT_flat_size, targets_dim)
    self.fcFinal = nn.Linear(targets_dim, targets_dim)


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
    adjT = asym_block(adjT, self.bn1, self.conv1, self.conv1_h, self.conv1_v)
    adjT = asym_block(adjT, self.bn2, self.conv2, self.conv2_h, self.conv2_v)
    adjT = asym_block(adjT, self.bn3, self.conv3, self.conv3_h, self.conv3_v)
    adjT = asym_block(adjT, self.bn4, self.conv4, self.conv4_h, self.conv4_v)
    return adjT.shape, prob.shape
        
  def forward(self, prob, adjT):
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
    adjT = asym_block(adjT, self.bn1, self.conv1, self.conv1_h, self.conv1_v)
    adjT = asym_block(adjT, self.bn2, self.conv2, self.conv2_h, self.conv2_v)
    adjT = asym_block(adjT, self.bn3, self.conv3, self.conv3_h, self.conv3_v)
    adjT = asym_block(adjT, self.bn4, self.conv4, self.conv4_h, self.conv4_v)

    # combine output of both branches
    x = self.combine(prob, adjT)
    x = F.relu(self.fcFinal(x))
    return x

  def dense_block(self, x, bn, conv, conv_h, conv_v):
    out = x
    x = F.relu(conv_v(x))
    x = F.relu(conv_h(x))
    x = torch.cat([out,x], dim=1)
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


if __name__ == "__main__":
  inputs_dim = 256
  targets_dim = 256
  encodings_dim = [8,8,8]
  net = AdjTAsymModel(inputs_dim=inputs_dim, targets_dim=targets_dim, encodings_dim=encodings_dim)
  print(net)

