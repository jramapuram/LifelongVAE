import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import shp
# from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils


def copy_layer(sess, src_layer, src_scope, dest_layer, dst_scope):
    src_vars = [v for v in tf.global_variables()
                if src_layer.scope in v.name and src_scope in v.name]
    dest_vars = [v for v in tf.global_variables()
                 if dest_layer.scope in v.name and dst_scope in v.name]

    copy_ops = []
    for s, d in zip(src_vars, dest_vars):
        if ('BatchNorm' not in s.name or 'BatchNorm' not in d.name) \
           and ('Adam' not in s.name or 'Adam' not in d.name):
            if s.get_shape().as_list() == d.get_shape().as_list():
                print 'copying %s [%s] --> %s [%s]' \
                    % (s.name, s.get_shape().as_list(),
                       d.name, d.get_shape().as_list())
                copy_ops.append(d.assign(s))

    sess.run(copy_ops)


def reinit_last_layer(sess, dest_layer):
    dst_proj_vars = [v for v in tf.global_variables()
                     if dest_layer.scope in v.name
                     and 'projection' in v.name]
    print 'proj_vars = ', dst_proj_vars

    reinit_ops = [d.initializer for d in dst_proj_vars]
    sess.run(reinit_ops)


def _get_normalizer(is_training, use_bn, use_ln):
    '''
    Helper to get normalizer function and params
    '''
    batch_norm_params = {'is_training': is_training,
                         'decay': 0.999, 'center': True,
                         'scale': True, 'updates_collections': None}
    layer_norm_params = {'center': True, 'scale': True}

    if use_ln:
        print 'using layer norm'
        normalizer_fn = slim.layer_norm
        normalizer_params = layer_norm_params
    elif use_bn:
        print 'using batch norm'
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        print 'not using any layer normalization scheme'
        normalizer_fn = None
        normalizer_params = None

    return [normalizer_fn, normalizer_params]


def forward(inputs, operator):
    '''
    Helper function to forward pass on the inputs using the provided model
    '''
    return operator.get_model(inputs)


class CNNEncoder(object):
    def __init__(self, sess, latent_size, is_training,
                 activation=tf.nn.elu, df_dim=32,
                 use_bn=False, use_ln=False,
                 scope="cnn_encoder"):
        self.sess = sess
        self.layer_type = "cnn"
        self.df_dim = df_dim
        self.latent_size = latent_size
        self.activation = activation
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.scope = scope
        self.is_training = is_training

    def get_info(self):
        return {'activation': self.activation.__name__,
                'latent_size': self.latent_size,
                'sizes': self.get_sizing(),
                'use_bn': str(self.use_bn),
                'use_ln': str(self.use_ln)}

    def get_sizing(self):
        return '4_5x5xN_s2_fc%d' % (self.latent_size)

    def get_detailed_sizing(self):
        return 's2_5x5x%d_' % self.df_dim \
            + 's2_5x5x%d_' % self.df_dim*2 \
            + 's2_5x5x%d_' % self.df_dim*4 \
            + 's2_5x5x%d_' % self.df_dim*8 \
            + 'fc%d' % self.latent_size

    def get_model(self, x):
        # get the normalizer function and parameters
        normalizer_fn, normalizer_params = _get_normalizer(self.is_training,
                                                           self.use_bn,
                                                           self.use_ln)

        winit = tf.contrib.layers.xavier_initializer_conv2d()
        # winit = tf.truncated_normal_initializer(stddev=0.01)
        with tf.variable_scope(self.scope):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=self.activation,
                                # weights_initializer=winit,
                                # biases_initializer=tf.constant_initializer(0),
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params):
                xshp = x.get_shape().as_list()
                x_flat = x if len(xshp) > 2 else tf.expand_dims(x, -1)

                # x_flat = tf.reshape(x, [-1, xshp[0], xshp[1],
                #                         xshp[2] if len(xshp) > 2 else 1])
                # h0 = slim.conv2d(x_flat, 32, [5, 5], stride=2)
                # h1 = slim.conv2d(h0, 64, [5, 5], stride=2)
                # h2 = slim.conv2d(h1, 128, [5, 5], stride=2, padding='VALID')
                # h2_flat = tf.reshape(h2, [xshp[0], -1])

                h0 = slim.conv2d(x_flat, self.df_dim, [5, 5], stride=1, padding='VALID')
                h1 = slim.conv2d(h0, self.df_dim*2, [4, 4], stride=2, padding='VALID')
                h2 = slim.conv2d(h1, self.df_dim*4, [4, 4], stride=1, padding='VALID')
                h3 = slim.conv2d(h2, self.df_dim*8, [4, 4], stride=2, padding='VALID')
                h4 = slim.conv2d(h3, self.df_dim*16, [4, 4], stride=1, padding='VALID')
                h5 = slim.conv2d(h4, self.df_dim*16, [1, 1], stride=1, padding='VALID')
                h6 = slim.conv2d(h5, self.latent_size, [1, 1], stride=1,
                                 normalizer_fn=None,
                                 activation_fn=None, padding='VALID')

                #return tf.reshape(h6, [-1, self.latent_size])
                return tf.reshape(h6, [xshp[0], -1])

                # h3_flat = tf.reshape(h3, [xshp[0], -1])

                # return slim.fully_connected(h3_flat,
                #                             self.latent_size,
                #                             normalizer_fn=None,
                #                             activation_fn=None)


class DenseEncoder(object):
    def __init__(self, sess, latent_size, is_training,
                 activation=tf.nn.elu,
                 sizes=[512, 512], use_bn=False, use_ln=False,
                 scope="dense_encoder"):
        self.sess = sess
        self.layer_type = "dnn"
        self.latent_size = latent_size
        self.activation = activation
        self.sizes = sizes
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.scope = scope
        self.is_training = is_training

    def get_info(self):
        return {'activation': self.activation.__name__,
                'latent_size': self.latent_size,
                'sizes': str(self.sizes),
                'use_bn': str(self.use_bn),
                'use_ln': str(self.use_ln)}

    def get_sizing(self):
        return str(self.sizes)

    def get_model(self, inputs):
        # get the normalizer function and parameters
        normalizer_fn, normalizer_params = _get_normalizer(self.is_training,
                                                           self.use_bn,
                                                           self.use_ln)

        winit = tf.contrib.layers.xavier_initializer()
        binit = tf.constant_initializer(0)
        with tf.variable_scope(self.scope):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=self.activation,
                                weights_initializer=winit,
                                biases_initializer=binit,
                                normalizer_fn=normalizer_fn,
                                normalizer_params=normalizer_params):
                    layers = slim.stack(inputs, slim.fully_connected,
                                        self.sizes, scope="layer")

            return slim.fully_connected(layers, self.latent_size,
                                        activation_fn=None,
                                        weights_initializer=winit,
                                        biases_initializer=binit,
                                        scope='projection')
