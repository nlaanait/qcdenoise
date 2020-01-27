import tensorflow as tf
import numpy as np

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

def train():
    pass
    


