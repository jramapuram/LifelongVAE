import tensorflow as tf
from utils import gumbel_softmax


def gaussian_reparmeterization(logits_z, eps=None):
    '''
    The vanilla gaussian reparameterization from Kingma et. al

    z = mu + sigma * N(0, I)
    '''
    zshp = logits_z.get_shape().as_list()
    assert zshp[1] % 2 == 0
    z_log_sigma_sq = logits_z[:, 0:zshp[0]/2]
    z_mean = logits_z[:, zshp[0]/2:]

    if eps is None:
        eps = tf.random_normal(tf.shape(z_mean), 0, 1,
                               dtype=tf.float32)

    cov = tf.mul(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)
    z = tf.add(z_mean, cov, name="z")

    kl = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean)
                              - tf.exp(z_log_sigma_sq), 1)
    return [z, kl]


def gumbel_reparmeterization(logits_z, tau, rnd_sample=None, eps=1e-9):
    '''
    The gumbel-softmax reparameterization
    '''

    latent_size = logits_z.get_shape().as_list()[1]
    q_z = tf.nn.softmax(logits_z)
    log_q_z = tf.log(q_z + eps)
    p_z = 1.0 / latent_size
    log_p_z = tf.log(p_z + eps)

    # set hard=True for ST Gumbel-Softmax
    z = tf.reshape(gumbel_softmax(logits_z, tau,
                                  hard=True,
                                  rnd_sample=rnd_sample),
                   [-1, latent_size])
    print 'z_internal = ', z.get_shape().as_list()

    # kl = tf.reshape(p_z * (log_p_z - log_q_z),
    #                 [-1, latent_size])
    kl = tf.reduce_sum(tf.reshape(q_z * (log_q_z - log_p_z),
                                  [-1, latent_size]), [1])
    return [z, kl]
