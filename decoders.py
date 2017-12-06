import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from encoders import _get_normalizer
from utils import shp


class CNNDecoder(object):
    def __init__(self, sess, input_size, is_training,
                 gf_dim=64, double_channels=False,
                 activation=tf.nn.elu, use_bn=False,
                 use_ln=False, scope="cnn_decoder"):
        self.sess = sess
        self.layer_type = "cnn"
        self.input_size = input_size
        self.gf_dim = gf_dim
        self.activation = activation
        self.double_channels = double_channels
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.scope = scope
        self.is_training = is_training

        # compute the sizing automatically
        # self.s_h, self.s_w, self.s_h2, self.s_w2, self.s_h4, \
        #     self.s_w4, self.s_h8, self.s_w8, self.s_h16, self.s_w16 \
        #     = self._compute_sizing()

    def get_info(self):
        return {'activation': self.activation.__name__,
                'sizes': self.get_sizing(),
                'use_bn': str(self.use_bn),
                'use_ln': str(self.use_ln)}

    def get_sizing(self):
        #return 'fc%d_4_5x5xN_s2' % (self.gf_dim*8*self.s_h16*self.s_w16)
        return 'cifar_decoder'

    def get_detailed_sizing(self):
        return 'fc%d_' % (self.gf_dim*8*self.s_h16*self.s_w16) \
            + 's2_5x5x%d_' % (self.gf_dim*4) \
            + 's2_5x5x%d_' % (self.gf_dim*2) \
            + 's2_5x5x%d_' % (self.gf_dim*1) \
            + 's2_5x5x%d_' % (self.gf_dim*1)
        # return str(self.gf_dim*8*self.s_h16*self.s_w16) \
        #     + str([self.s_h8, self.s_w8, self.gf_dim*4]) \
        #     + str([self.s_h4, self.s_w4, self.gf_dim*2]) \
        #     + str([self.s_h2, self.s_w2, self.gf_dim]) \
        #     + str([self.s_h, self.s_w, self.input_size[-1]])

    @staticmethod
    def conv_out_size_same(size, stride):
        '''From DCGAN (carpedm20)'''
        return int(math.ceil(float(size) / float(stride)))

    def _compute_sizing(self):
        s_h, s_w = self.input_size[0], self.input_size[1]
        s_h2, s_w2 = [self.conv_out_size_same(s_h, 2),
                      self.conv_out_size_same(s_w, 2)]
        s_h4, s_w4 = [self.conv_out_size_same(s_h2, 2),
                      self.conv_out_size_same(s_w2, 2)]
        s_h8, s_w8 = [self.conv_out_size_same(s_h4, 2),
                      self.conv_out_size_same(s_w4, 2)]
        s_h16, s_w16 = [self.conv_out_size_same(s_h8, 2),
                        self.conv_out_size_same(s_w8, 2)]

        return [s_h, s_w, s_h2, s_w2, s_h4,
                s_w4, s_h8, s_w8, s_h16, s_w16]

    def get_model(self, z):
        # get the normalizer function and parameters
        normalizer_fn, normalizer_params = _get_normalizer(self.is_training,
                                                           self.use_bn,
                                                           self.use_ln)

        # winit = tf.contrib.layers.xavier_initializer_conv2d()
        winit = tf.random_normal_initializer(stddev=0.02)

        with tf.variable_scope(self.scope):
            with slim.arg_scope([slim.conv2d_transpose],
                                activation_fn=self.activation,
                                weights_initializer=winit,
                                biases_initializer=tf.constant_initializer(0),
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params):

                z_exp = tf.reshape(z, [-1, 1, 1, shp(z)[-1]])
                h0 = slim.conv2d_transpose(z_exp, num_outputs=self.gf_dim*8,
                                           kernel_size=[4, 4],
                                           stride=1, padding='VALID')
                h1 = slim.conv2d_transpose(h0, num_outputs=self.gf_dim*4,
                                           kernel_size=[4, 4],
                                           stride=2, padding='VALID')
                h2 = slim.conv2d_transpose(h1, num_outputs=self.gf_dim*2,
                                           kernel_size=[4, 4],
                                           stride=1, padding='VALID')
                h3 = slim.conv2d_transpose(h2, num_outputs=self.gf_dim,
                                           kernel_size=[4, 4],
                                           stride=2, padding='VALID')
                h4 = slim.conv2d_transpose(h3, num_outputs=self.gf_dim,
                                           kernel_size=[5, 5],
                                           stride=1, padding='VALID')
                channels = self.input_size[-1] if len(self.input_size) > 2 else 1
                if self.double_channels:
                    channels *= 2

            final = slim.conv2d(h4, channels, [1, 1], stride=1,
                                activation_fn=None, normalizer_fn=None,
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                biases_initializer=tf.constant_initializer(0),
                                padding='VALID')
            print('conv decoded final = ', final.get_shape().as_list())
            return final

                # proj_z = slim.fully_connected(z, self.gf_dim*8*self.s_h16*self.s_w16,
                #                               activation_fn=None,
                #                               weights_initializer=tf.contrib.layers.xavier_initializer(),
                #                               biases_initializer=tf.constant_initializer(0),
                #                               scope='proj_z')
                # z_flat = self.activation(tf.reshape(proj_z, [-1, self.s_h16,
                #                                              self.s_w16,
                #                                              self.gf_dim*8]))
                # h0 = slim.conv2d_transpose(z_flat, num_outputs=self.gf_dim*4,
                #                            kernel_size=[5, 5],
                #                            stride=[2, 2],
                #                            padding='SAME')
                # h1 = slim.conv2d_transpose(h0, num_outputs=self.gf_dim*2,
                #                            kernel_size=[5, 5],
                #                            stride=[2, 2],
                #                            padding='SAME')
                # h2 = slim.conv2d_transpose(h1, num_outputs=self.gf_dim,
                #                            kernel_size=[5, 5],
                #                            stride=[2, 2],
                #                            padding='SAME')
                # channels = self.input_size[-1] if len(self.input_size) > 2 else 1
                # if self.double_channels:
                #     channels *= 2

                # h3 = slim.conv2d_transpose(h2, num_outputs=channels,
                #                            kernel_size=[5, 5],
                #                            stride=[2, 2],
                #                            padding='SAME',
                #                            normalizer_fn=None,
                #                            normalizer_params=None,
                #                            activation_fn=None)
                # print 'h3 = ', h3.get_shape().as_list()
                # return slim.flatten(h3) if channels == 1 else h3
