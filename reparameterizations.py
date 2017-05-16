import tensorflow as tf
import tensorflow.contrib.distributions as d

from utils import gumbel_softmax


# def gaussian_reparmeterization(logits_z, rnd_sample=None):
#     '''
#     The vanilla gaussian reparameterization from Kingma et. al

#     z = mu + sigma * N(0, I)
#     '''
#     zshp = logits_z.get_shape().as_list()
#     assert zshp[1] % 2 == 0
#     z_log_sigma_sq = logits_z[:, 0:zshp[1]/2]
#     z_mean = logits_z[:, zshp[1]/2:]

#     z = d.MultivariateNormalDiagWithSoftplusScale(z_mean + 1e-9,
#                                                   z_log_sigma_sq)
#     prior = d.MultivariateNormalDiagWithSoftplusScale(tf.zeros_like(z_mean) + 1e-9,
#                                                       tf.ones_like(z_log_sigma_sq))

#     return [z.sample(), d.kl(prior, z, allow_nan_stats=False)]


def gaussian_reparmeterization(logits_z, rnd_sample=None):
    '''
    The vanilla gaussian reparameterization from Kingma et. al

    z = mu + sigma * N(0, I)
    '''
    zshp = logits_z.get_shape().as_list()
    assert zshp[1] % 2 == 0
    z_log_sigma_sq = logits_z[:, 0:zshp[1]/2]
    z_mean = logits_z[:, zshp[1]/2:]
    print 'zmean shp = ', z_mean.get_shape().as_list()
    print 'z_log_sigma_sq shp = ', z_log_sigma_sq.get_shape().as_list()

    if rnd_sample is None:
        rnd_sample = tf.random_normal(tf.shape(z_mean), 0, 1,
                                      dtype=tf.float32)

    # cov = tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), rnd_sample)
    # softplus = log(exp(features) + 1)
    cov = tf.multiply(tf.sqrt(tf.nn.softplus(z_log_sigma_sq)), rnd_sample)
    z = tf.add(z_mean, cov, name="z")

    reduce_index = [1] if len(zshp) == 2 else [1, 2]
    kl = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean)
                              - tf.nn.softplus(z_log_sigma_sq), reduce_index)
    # kl = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean)
    #                           - tf.exp(z_log_sigma_sq), reduce_index)
    return [z, kl]


def gumbel_reparmeterization(logits_z, tau, rnd_sample=None,
                             hard=True, eps=1e-9):
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
                                  hard=hard,
                                  rnd_sample=rnd_sample),
                   [-1, latent_size])
    print 'z_gumbel = ', z.get_shape().as_list()

    # kl = tf.reshape(p_z * (log_p_z - log_q_z),
    #                 [-1, latent_size])
    reduce_index = [1] if len(logits_z.get_shape().as_list()) == 2 else [1, 2]
    kl = tf.reduce_sum(tf.reshape(q_z * (log_q_z - log_p_z),
                                  [-1, latent_size]), reduce_index)
    return [z, kl]
