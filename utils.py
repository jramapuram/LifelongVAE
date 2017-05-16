import os
import re
import string
import math
import random
import hashlib
import tarfile
import tensorflow as tf
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from copy import deepcopy
from tensorflow.python.framework import ops
from sklearn.preprocessing import MinMaxScaler


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BatchBuffer(object):
    def __init__(self, func, batch_size, buffer_size=10):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.func = func
        self.buffer = func(batch_size*buffer_size)
        self.remaining = buffer_size

    def get(self):
        if self.remaining == 0:
            self.buffer = self.func(self.batch_size*self.buffer_size)
            self.remaining = self.buffer_size

        start = len(self.buffer[0]) - self.remaining * self.batch_size
        end = start + self.batch_size
        retval = self.buffer[0][start:end], self.buffer[1][start:end]
        self.remaining -= 1
        return retval


def random_str(length):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(length))


def find_latest_file(path, regex):
    '''
    Given a path and a regex string return the latest filename
    Eg: r'vae(\d)+' will return vae(latestnumber)_...
    TODO: this can probably be optimized to sort and start from the back, but meh
    '''
    files_in_dir = os.listdir(path)
    latest = (None, 0)
    for f in files_in_dir:
        print 'file = ', f, ' | rgx = ', regex
        r = re.match(regex, f)
        current_index = int(r.group(1)) if r else 0
        print current_index, latest[1], current_index >= latest[1]
        latest = (f, current_index) if current_index >= latest[1] else latest

    return latest


def linear(input_, output_size, activation=None, scope=None,
           bias_init=0.0, with_params=False, reuse=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear", reuse=reuse):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_init))

        if activation is None:
            result = tf.matmul(input_, matrix) + bias
        else:
            result = activation(tf.matmul(input_, matrix) + bias)

        return [result, matrix, bias] if with_params else result


def hash_list(str_list):
    h = hashlib.new('ripemd160')
    for val in str_list:
        h.update(str(val))

    return h.hexdigest()


def unit_scale(t):
    m, v = tf.nn.moments(t, axes=[0])
    return (t - m)/(v + 1e-9)


# returns a matrix with [num_rows, num_cols] where the indices value is 1
def one_hot(num_cols, indices):
    num_rows = len(indices)
    mat = np.zeros((num_rows, num_cols))
    mat[np.arange(num_rows), indices] = 1
    return mat


# 2x2 conv [downsample]
# F = [2, 2, 1, 32] w/ S=[1,2,2,1]
def conv_relu_2x2(x, W, b, filter_size, is_training, scope='bn'):
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME',
                        use_cudnn_on_gpu=True) + b
    # bn = batch_norm(conv, filter_size, is_training, scope=scope)
    # return tf.nn.relu(bn)
    return tf.nn.relu(conv)


# 1x1 conv [shape preserving]
def conv_relu_1x1(x, W, b, filter_size, is_training, scope='bn'):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',
                        use_cudnn_on_gpu=True) + b
    # bn = batch_norm(conv, filter_size, is_training, scope=scope)
    # return tf.nn.relu(bn)
    return tf.nn.relu(conv)


# 2d max pool [downsample]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 1d max pool [downsample]
def max_pool_1x2(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')


# a debug helper to print out the tensor
def tensor_printer(tensor, name='tensor', summary_len=100):
    summary_str = name + ' = '
    return tf.Print(tensor, [tensor], summary_str, summarize=summary_len)


# filters labels by removing items that are in blacklist
# returns the tuple of images, labels that match filtration above
def zip_filter_unzip(images, labels, blacklist):
    return zip(*([im, lbl]
                 for im, lbl in zip(images, labels)
                 if lbl not in blacklist))


def compress(output_file, sources):
    tar = tarfile.open(output_file, "w:gz")
    if type(sources) != list:
        sources = [sources]

    for name in sources:
        tar.add(name)

    tar.close()


def shp(tensor):
    return tensor.get_shape().as_list()


def shuffle_jointly(*args):
    '''
    accepts n args, concatinates them all together
    and then shuffles along batch_dim and returns them unsplit
    '''
    shps = [a.get_shape().as_list()[-1] for a in args]
    concated = tf.random_shuffle(tf.concat(values=args, axis=1))
    splits = []
    current_max = 0
    for begin in shps:
        splits.append(concated[:, current_max:current_max + begin])
        current_max += begin

    return splits


def shuffle_rows_based_on_indices(row_indices, *args):
    num_cols = 0
    for arg in args:
        tensor_shape = arg.get_shape().as_list()
        assert len(tensor_shape) == 2, "provide 2d tensors"
        num_cols += tensor_shape[1]

    indices = [[int(x), int(y)] for x in row_indices
               for y in range(num_cols)]
    return shuffle_based_on_indices(indices, 1, *args)


def shuffle_cols_based_on_indices(col_indices, *args):
    num_rows = 0
    for arg in args:
        tensor_shape = arg.get_shape().as_list()
        assert len(tensor_shape) == 2, "provide 2d tensors"
        num_rows += tensor_shape[0]

    indices = [[int(x), int(y)] for x in range(num_rows)
               for y in col_indices]
    return shuffle_based_on_indices(indices, 0, *args)


def shuffle_based_on_indices(indices, index, *args):
    '''
    Generic mass shuffler that utilizes the indices to shuffle the parameters
    It then returns the individual arrays split apart
    '''
    shps = [a.get_shape().as_list()[index] for a in args]
    concated = tf.concat(axis=index, values=args) if len(args) > 1 else args[0]
    cshp = concated.get_shape().as_list()

    shuffled = tf.reshape(tf.gather_nd(concated, indices), cshp)
    splits = []
    current_max = 0
    for begin in shps:
        if index == 0:
            splits.append(shuffled[current_max:current_max + begin, :])
        elif index == 1:
            splits.append(shuffled[:, current_max:current_max + begin])
        elif index == 2:
            splits.append(shuffled[:, :, current_max:current_max + begin])
        else:
            raise Exception("invalid shuffle dim")

        current_max += begin

    return splits if len(args) > 1 else splits[0]


def min_max_normalize(X, min=0.0, max=1.0):
    '''
    The transformation is given by::
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min
    where min, max = feature_range.
    '''
    x_std = (X - tf.reduce_min(X, axis=0)) / (tf.reduce_max(X, axis=0)
                                              - tf.reduce_min(X, axis=0))
    return x_std * (max - min) + min


def write_csv(arr, base_dir, filename):
    with open("%s/%s" % (base_dir, filename), 'a') as f:
        np.savetxt(f, arr, delimiter=",")


def normalize(x, scale_range=True):
    if len(x.shape) == 2:
        denominator = np.expand_dims(np.clip(np.std(x, axis=1), 1e-9, 1e25), 1)
        cleaned = (x - np.expand_dims(np.mean(x, axis=1), 1)) / denominator
    elif len(x.shape) == 1:
        cleaned = (x - np.mean(x)) / np.clip(np.std(x), 1e-9, 1e25)
    else:
        raise Exception("Unknown shape provided")

    return MinMaxScaler().fit_transform(cleaned) if scale_range else cleaned


def save_fig(m, name, reshaped=[28, 28]):
    plt.figure()
    plt.imshow(m.reshape(reshaped))
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def generate_random_categorical(num_targets, batch_size):
    ''' Helper to return a categorical of [batch_size, num_targets]
        where one of the num_targets are chosen at random[uniformly]'''
    indices = tf.random_uniform([batch_size], minval=0,
                                maxval=num_targets, dtype=tf.int32)
    return tf.one_hot(indices, num_targets, dtype=tf.float32)


def find_top_K(hist, num_top=1):
    values, indices = tf.nn.top_k(hist, num_top, sorted=True)
    return indices


def tf_mean_std_normalize(x, eps=1e-20):
    assert len(x.get_shape().as_list()) > 1
    u, var = tf.nn.moments(x, axes=[1])
    return (x - tf.expand_dims(u, 1)) / (tf.expand_dims(var, 1) + eps)


def tf_scale_unit_range(tensor):
    return tf.div(
        tf.sub(
            tensor,
            tf.reduce_min(tensor)
        ),
        tf.sub(
            tf.reduce_max(tensor),
            tf.reduce_min(tensor)
        )
    )


def tf_normalize(x, eps=1e-20):
    assert len(x.get_shape().as_list()) > 1
    return x / tf.expand_dims(tf.reduce_sum(x, axis=1) + eps, axis=1)


# From blog.evjang.com
def sample_gumbel(shape, maxval=1, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=maxval)
    return -tf.log(-tf.log(U + eps) + eps)


# From blog.evjang.com
def gumbel_softmax_sample(logits, temperature, rnd_sample=None):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    if rnd_sample is None:
        y = logits + sample_gumbel(tf.shape(logits))
    else:
        y = logits + rnd_sample

    return tf.nn.softmax(y / temperature)


# From blog.evjang.com
def gumbel_softmax(logits, temperature, hard=False, rnd_sample=None):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classesk
    """
    y = gumbel_softmax_sample(logits, temperature, rnd_sample)
    if hard:
        # k = tf.shape(logits)[-1]
        # y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)

        # Basically says: y_hard = (y == max(y, axis=1))
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y  # haxx

    return y
