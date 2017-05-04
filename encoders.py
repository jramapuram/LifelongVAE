import tensorflow as tf
import tensorflow.contrib.slim as slim


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
                 activation=tf.nn.elu,
                 use_bn=False, use_ln=False,
                 activate_last_layer=False):
        self.sess = sess
        self.layer_type = "cnn"
        self.latent_size = latent_size
        self.activation = activation
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.is_training = is_training
        self.activate_last_layer = activate_last_layer

    def get_info(self):
        return {'activation': self.activation.__name__,
                'latent_size': self.latent_size,
                'sizes': str(['32x5x5', '64x5x5', '128x5x5',
                              str(self.latent_size)]),
                'use_bn': str(self.use_bn),
                'use_ln': str(self.use_ln),
                'activ_last_layer': str(self.activate_last_layer)}

    def get_sizing(self):
        return str(['32x5x5', '64x5x5', '128x5x5',
                    str(self.latent_size)])

    def get_model(self, x):
        # get the normalizer function and parameters
        normalizer_fn, normalizer_params = _get_normalizer(self.is_training,
                                                           self.use_bn,
                                                           self.use_ln)

        winit = tf.contrib.layers.xavier_initializer_conv2d()
        # winit = tf.truncated_normal_initializer(stddev=0.01)
        with slim.arg_scope([slim.conv2d],
                            activation_fn=self.activation,
                            weights_initializer=winit,
                            biases_initializer=tf.constant_initializer(0),
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params):
            xshp = x.get_shape().as_list()
            x_flat = tf.reshape(x, [-1, xshp[0], xshp[1], 1])
            h0 = slim.conv2d(x_flat, 32, [5, 5], stride=2)
            h1 = slim.conv2d(h0, 64, [5, 5], stride=2)
            #h2 = slim.conv2d(h1, 128, [5, 5], stride=2)
            h2_flat = tf.reshape(h1, [xshp[0], -1])

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.constant_initializer(0)):
            if self.activate_last_layer:
                return slim.fully_connected(h2_flat, self.latent_size,
                                            activation_fn=self.activation)
            else:
                return slim.fully_connected(h2_flat, self.latent_size,
                                            activation_fn=None)


class DenseEncoder(object):
    def __init__(self, sess, latent_size, is_training,
                 activation=tf.nn.elu,
                 sizes=[512, 512], use_bn=False, use_ln=False,
                 activate_last_layer=False):
        self.sess = sess
        self.layer_type = "dnn"
        self.latent_size = latent_size
        self.activation = activation
        self.sizes = sizes
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.is_training = is_training
        self.activate_last_layer = activate_last_layer

    def get_info(self):
        return {'activation': self.activation.__name__,
                'latent_size': self.latent_size,
                'sizes': str(self.sizes),
                'use_bn': str(self.use_bn),
                'use_ln': str(self.use_ln),
                'activ_last_layer': str(self.activate_last_layer)}

    def get_sizing(self):
        return str(self.sizes)

    def get_model(self, inputs):
        # get the normalizer function and parameters
        normalizer_fn, normalizer_params = _get_normalizer(self.is_training,
                                                           self.use_bn,
                                                           self.use_ln)

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=self.activation,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.constant_initializer(0),
                            normalizer_fn=normalizer_fn,
                            normalizer_params=normalizer_params):
            if self.activate_last_layer:
                return slim.stack(inputs, slim.fully_connected, self.sizes, scope="layer")
            else:
                layers = slim.stack(inputs, slim.fully_connected, self.sizes, scope="layer")

        return slim.fully_connected(layers, self.latent_size, activation_fn=None,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.constant_initializer(0))
