import tensorflow as tf
import tensorflow.contrib.distributions as d

from utils import gumbel_softmax, shp


sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor


def gaussian_reparmeterization(logits_z, rnd_sample=None):
    '''
    The vanilla gaussian reparameterization from Kingma et. al

    z = mu + sigma * N(0, I)
    '''
    zshp = logits_z.get_shape().as_list()
    assert zshp[1] % 2 == 0
    q_sigma = 1e-6 + tf.nn.softplus(logits_z[:, 0:zshp[1]/2])
    q_mu = logits_z[:, zshp[1]/2:]

    # Prior
    p_z = d.Normal(loc=tf.zeros(zshp[1] / 2),
                   scale=tf.ones(zshp[1] / 2))

    with st.value_type(st.SampleValue()):
        q_z = st.StochasticTensor(d.Normal(loc=q_mu, scale=q_sigma))

    reduce_index = [1] if len(zshp) == 2 else [1, 2]
    kl = d.kl_divergence(q_z.distribution, p_z, allow_nan_stats=False)
    return [q_z, tf.reduce_sum(kl, reduce_index)]

# def gaussian_reparmeterization(logits_z, rnd_sample=None):
#     '''
#     The vanilla gaussian reparameterization from Kingma et. al

#     z = mu + sigma * N(0, I)
#     '''
#     zshp = logits_z.get_shape().as_list()
#     assert zshp[1] % 2 == 0
#     z_log_sigma_sq = logits_z[:, 0:zshp[1]/2]
#     z_mean = logits_z[:, zshp[1]/2:]
#     print 'zmean shp = ', z_mean.get_shape().as_list()
#     print 'z_log_sigma_sq shp = ', z_log_sigma_sq.get_shape().as_list()

#     if rnd_sample is None:
#         rnd_sample = tf.random_normal(tf.shape(z_mean), 0, 1,
#                                       dtype=tf.float32)

#     # cov = tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), rnd_sample)
#     # softplus = log(exp(features) + 1)
#     cov = tf.multiply(tf.sqrt(tf.nn.softplus(z_log_sigma_sq)), rnd_sample)
#     z = tf.add(z_mean, cov, name="z")

#     reduce_index = [1] if len(zshp) == 2 else [1, 2]
#     kl = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean)
#                               - tf.nn.softplus(z_log_sigma_sq), reduce_index)
#     # kl = -0.5 * tf.reduce_sum(1.0 + z_log_sigma_sq - tf.square(z_mean)
#     #                           - tf.exp(z_log_sigma_sq), reduce_index)
#     return [z, kl]


def gumbel_reparmeterization(logits_z, tau, rnd_sample=None,
                             hard=True, eps=1e-9):
    '''
    The gumbel-softmax reparameterization
    '''
    latent_size = logits_z.get_shape().as_list()[1]

    # Prior
    p_z = d.OneHotCategorical(probs=tf.constant(1.0/latent_size,
                                                shape=[latent_size]))
    # p_z = d.RelaxedOneHotCategorical(probs=tf.constant(1.0/latent_size,
    #                                                    shape=[latent_size]),
    #                                  temperature=10.0)
    # p_z = 1.0 / latent_size
    # log_p_z = tf.log(p_z + eps)

    with st.value_type(st.SampleValue()):
        q_z = st.StochasticTensor(d.RelaxedOneHotCategorical(temperature=tau,
                                                             logits=logits_z))
        q_z_full = st.StochasticTensor(d.OneHotCategorical(logits=logits_z))

    reduce_index = [1] if len(logits_z.get_shape().as_list()) == 2 else [1, 2]
    kl = d.kl_divergence(q_z_full.distribution, p_z, allow_nan_stats=False)
    if len(shp(kl)) > 1:
        return [q_z, tf.reduce_sum(kl, reduce_index)]
    else:
        return [q_z, kl]

    # reduce_index = [1] if len(logits_z.get_shape().as_list()) == 2 else [1, 2]
    # kl = tf.reduce_sum(tf.reshape(q_z.distribu * (log_q_z - p_z. log_p_z),
    #                               [-1, latent_size]), reduce_index)
    # return [z, kl]



# def gumbel_reparmeterization(logits_z, tau, rnd_sample=None,
#                              hard=True, eps=1e-9):
#     '''
#     The gumbel-softmax reparameterization
#     '''

#     latent_size = logits_z.get_shape().as_list()[1]
#     q_z = tf.nn.softmax(logits_z)
#     log_q_z = tf.log(q_z + eps)
#     p_z = 1.0 / latent_size
#     log_p_z = tf.log(p_z + eps)

#     # set hard=True for ST Gumbel-Softmax
#     z = tf.reshape(gumbel_softmax(logits_z, tau,
#                                   hard=hard,
#                                   rnd_sample=rnd_sample),
#                    [-1, latent_size])
#     print 'z_gumbel = ', z.get_shape().as_list()

#     # kl = tf.reshape(p_z * (log_p_z - log_q_z),
#     #                 [-1, latent_size])
#     reduce_index = [1] if len(logits_z.get_shape().as_list()) == 2 else [1, 2]
#     kl = tf.reduce_sum(tf.reshape(q_z * (log_q_z - log_p_z),
#                                   [-1, latent_size]), reduce_index)
#     return [z, kl]
