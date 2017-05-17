import numpy as np
import tensorflow as tf
from tf_utils.layers import conv2d, deconv2d, ar_multiconv2d, \
    resize_nearest_neighbor
from tensorflow.contrib.framework.python.ops import arg_scope


def _split(x, split_dim, split_sizes):
    n = len(list(x.get_shape()))
    dim_size = np.sum(split_sizes)
    assert int(x.get_shape()[split_dim]) == dim_size
    ids = np.cumsum([0] + split_sizes)
    ids[-1] = -1
    begin_ids = ids[:-1]

    ret = []
    for i in range(len(split_sizes)):
        cur_begin = np.zeros([n], dtype=np.int32)
        cur_begin[split_dim] = begin_ids[i]
        cur_end = np.zeros([n], dtype=np.int32) - 1
        cur_end[split_dim] = split_sizes[i]
        ret += [tf.slice(x, cur_begin, cur_end)]
    return ret


def gaussian_diag_logps(mean, logvar, sample=None):
    if sample is None:
        noise = tf.random_normal(tf.shape(mean))
        sample = mean + tf.exp(0.5 * logvar) * noise

    return -0.5 * (np.log(2 * np.pi) + logvar
                   + tf.square(sample - mean) / tf.exp(logvar))


class DiagonalGaussian(object):
    def __init__(self, mean, logvar, sample=None):
        self.mean = mean
        self.logvar = logvar

        if sample is None:
            noise = tf.random_normal(tf.shape(mean))
            sample = mean + tf.exp(0.5 * logvar) * noise
        self.sample = sample

    def logps(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)


class IAFLayer(object):
    def __init__(self, mode, downsample,
                 latent_size, batch_size,
                 num_importance_samples,
                 kl_min,
                 activation=tf.nn.elu):
        self.mode = mode
        self.downsample = downsample
        self.activation = activation
        self.kl_min = kl_min
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.num_importance_samples = num_importance_samples

    def up(self, input, **_):
        # h_size = hps.h_size
        h_size = input.get_shape().as_list()[1]
        z_size = self.latent_size
        stride = [2, 2] if self.downsample else [1, 1]

        with arg_scope([conv2d]):
            x = tf.nn.elu(input)
            x = conv2d("up_conv1", x, 2 * z_size + 2 * h_size, stride=stride)
            self.qz_mean, self.qz_logsd, self.up_context, h \
                = _split(x, 1, [z_size, z_size, h_size, h_size])

            h = tf.nn.elu(h)
            h = conv2d("up_conv3", h, h_size)
            if self.downsample:
                input = resize_nearest_neighbor(input, 0.5)
            return input + 0.1 * h

    def down(self, input):
        # h_size = hps.h_size
        h_size = input.get_shape().as_list()[1]
        z_size = self.latent_size

        with arg_scope([conv2d, ar_multiconv2d]):
            x = tf.nn.elu(input)
            x = conv2d("down_conv1", x, 4 * z_size + h_size * 2)
            pz_mean, pz_logsd, rz_mean, rz_logsd, down_context, h_det \
                = _split(x, 1, [z_size] * 4 + [h_size] * 2)

            prior = DiagonalGaussian(pz_mean, 2 * pz_logsd)
            posterior = DiagonalGaussian(rz_mean + self.qz_mean,
                                         2 * (rz_logsd + self.qz_logsd))
            context = self.up_context + down_context

            if self.mode in ["init", "sample"]:
                z = prior.sample
            else:
                z = posterior.sample

            if self.mode == "sample":
                kl_cost = kl_obj = tf.zeros([self.batch_size
                                             * self.num_importance_samples])
            else:
                logqs = posterior.logps(z)
                x = ar_multiconv2d("ar_multiconv2d", z, context,
                                   [h_size, h_size], [z_size, z_size])
                arw_mean, arw_logsd = x[0] * 0.1, x[1] * 0.1
                z = (z - arw_mean) / tf.exp(arw_logsd)
                logqs += arw_logsd
                logps = prior.logps(z)

                kl_cost = logqs - logps

                if self.kl_min > 0:
                    # [0, 1, 2, 3] -> [0, 1] -> [1] / (b * k)
                    kl_ave = tf.reduce_mean(tf.reduce_sum(kl_cost, [2, 3]),
                                            [0], keep_dims=True)
                    kl_ave = tf.maximum(kl_ave, self.kl_min)
                    kl_ave = tf.tile(kl_ave, [self.batch_size *
                                              self.num_importance_samples, 1])
                    kl_obj = tf.reduce_sum(kl_ave, [1])
                else:
                    kl_obj = tf.reduce_sum(kl_cost, [1, 2, 3])
                kl_cost = tf.reduce_sum(kl_cost, [1, 2, 3])

            h = tf.concat(axis=1, values=[z, h_det])
            h = tf.nn.elu(h)
            if self.downsample:
                input = resize_nearest_neighbor(input, 2)
                h = deconv2d("down_deconv2", h, h_size)
            else:
                h = conv2d("down_conv2", h, h_size)
            output = input + 0.1 * h
            return output, kl_obj, kl_cost
