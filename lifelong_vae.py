import os
import sys
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as distributions
# from tensorflow.python.training.moving_averages import weighted_moving_average
from mnist_number import MNIST_Number, full_mnist
from utils import *
from reparameterizations import *
from encoders import forward, DenseEncoder, CNNEncoder
from decoders import CNNDecoder

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor
sys.setrecursionlimit(200)

flags = tf.flags
flags.DEFINE_string("target_dataset", "mnist", "mnist or regression.")
flags.DEFINE_string("model_type", "gan", "gan or vae")
flags.DEFINE_bool("sequential", 0, "sequential or not")
flags.DEFINE_integer("latent_size", 20, "Number of latent variables.")
flags.DEFINE_integer("epochs", 100, "Maximum number of epochs.")
flags.DEFINE_integer("batch_size", 100, "Mini-batch size for data subsampling.")
flags.DEFINE_integer("min_interval", 3000, "Minimum interval for specific dataset.")
flags.DEFINE_string("device", "/gpu:0", "Compute device.")
flags.DEFINE_boolean("allow_soft_placement", True, "Soft device placement.")
flags.DEFINE_float("device_percentage", "0.3", "Amount of memory to use on device.")
flags.DEFINE_string("use_ln", "none", "encoder / decoder / encoder_decoder for layer norm")
flags.DEFINE_string("use_bn", "none", "encoder / decoder / encoder_decoder for batch norm")
# flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout keep probability")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
FLAGS = flags.FLAGS

# Global variables
GLOBAL_ITER = 0  # keeps track of the iteration ACROSS models
TRAIN_ITER = 0  # the iteration of the current model


class VAE(object):
    """ Online Variational Autoencoder with consistent sampling.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling
    for more details on the original work.
    """
    def __init__(self, sess, input_size, batch_size, latent_size,
                 encoder, decoder, activation=tf.nn.softplus,
                 learning_rate=1e-3, submodel=0, vae_tm1=None):
        self.activation = activation
        self.learning_rate = learning_rate
        self.encoder_model = encoder
        self.decoder_model = decoder
        self.vae_tm1 = vae_tm1
        self.global_iter_base = GLOBAL_ITER
        self.input_size = input_size
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.iteration = 0
        self.submodel = submodel
        # self.num_discrete = self.submodel - 1  # TBD
        self.num_discrete = self.submodel + 1  # TODO: add dupe detection

        # gumbel params
        self.tau0 = 1.0
        self.tau_host = self.tau0
        self.anneal_rate = 0.00003
        # self.anneal_rate = 0.0003 #1e-5
        self.min_temp = 0.5

        # sess & graph
        self.sess = sess
        # self.graph = tf.Graph()

        # create these in scope
        self._create_variables()

        # Create autoencoder network
        self._create_network()

        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Create all the summaries and their corresponding ops
        self._create_summaries()

        # Check for NaN's
        # self.check_op = tf.add_check_numerics_ops()

        # collect variables & build saver
        self.vae_vars = [v for v in tf.global_variables()
                         if v.name.startswith(self.get_name())]
        self.saver = tf.train.Saver(tf.global_variables())  # XXX: use local

        self.init_op = tf.variables_initializer(self.vae_vars)
        print 'model: ', self.get_name()
        print 'there are ', len(self.vae_vars), ' vars in ', \
            tf.get_variable_scope().name, ' out of a total of ', \
            len(tf.global_variables()), ' with %d total trainable vars' \
            % len(tf.trainable_variables())

    def _create_variables(self):
        with tf.variable_scope(self.get_name()):
            # Create the placeholders if we are at the first model
            # Else simply pull the references
            if self.submodel == 0:
                self.is_training = tf.placeholder(tf.bool, name="is_training")
                self.x = tf.placeholder(tf.float32, shape=[self.batch_size,
                                                           self.input_size],
                                        name="input_placeholder")
            else:
                self.is_training = self.vae_tm1.is_training
                self.x = self.vae_tm1.x

            # gpu iteration count
            self.iteration_gpu = tf.Variable(0.0, trainable=False)
            self.iteration_gpu_op = self.iteration_gpu.assign_add(1.0)

            # gumbel related
            self.tau = tf.Variable(5.0, trainable=False, dtype=tf.float32,
                                   name="temperature")
            # self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)

    '''
    A helper function to create all the summaries.
    Adds things like image_summary, histogram_summary, etc.
    '''
    def _create_summaries(self):
        # Summaries and saver
        summaries = [tf.summary.scalar("vae_loss_mean", self.cost_mean),
                     tf.summary.scalar("vae_latent_loss_mean", self.latent_loss_mean),
                     tf.summary.histogram("vae_kl_unused", self.kl_unused_classes),
                     tf.summary.histogram("vae_kl_used", self.kl_used_classes),
                     tf.summary.histogram("vae_latent_dist", self.latent_kl),
                     tf.summary.scalar("vae_latent_loss_max", tf.reduce_max(self.latent_kl)),
                     tf.summary.scalar("vae_latent_loss_min", tf.reduce_min(self.latent_kl)),
                     tf.summary.scalar("vae_reconstr_loss_mean", self.reconstr_loss_mean),
                     tf.summary.scalar("vae_reconstr_loss_max", tf.reduce_max(self.reconstr_loss)),
                     tf.summary.scalar("vae_reconstr_loss_min", tf.reduce_min(self.reconstr_loss)),
                     tf.summary.histogram("z_dist", self.z)]

        # Display image summaries : i.e. samples from P(X|Z=z_i)
        # Visualize:
        #           1) augmented images;
        #           2) original images[current distribution]
        #           3) reconstructed images
        x_orig, x_aug, x_reconstr = shuffle_jointly(self.x, self.x_augmented, # noqa
                                                    self.x_reconstr_mean_activ)
        img_shp = [self.batch_size, 28, 28, 1]
        summaries += [tf.summary.image("x_augmented_t", tf.reshape(x_aug, img_shp), # noqa
                                       max_outputs=self.batch_size),
                      tf.summary.image("x_t", tf.reshape(x_orig, img_shp),
                                       max_outputs=self.batch_size),
                      tf.summary.image("x_reconstr_mean_activ_t",
                                       tf.reshape(x_reconstr, img_shp),
                                       max_outputs=self.batch_size)]

        # In addition show the following if they exist:
        #          4) Images from previous interval
        #          5) Distilled KL Divergence
        if hasattr(self, 'xhat_tm1'):
            num_xhat_tm1 = self.xhat_tm1.get_shape().as_list()
            img_shp = [-1, 28, 28, 1]
            summaries += [tf.summary.image("xhat_tm1",
                                           tf.reshape(self.xhat_tm1, img_shp),
                                           max_outputs=num_xhat_tm1[0]),
                          tf.summary.scalar("vae_kl_distill_mean",
                                            tf.reduce_mean(self.kl_consistency)),
                          tf.summary.scalar("kl_scaling_dist_min",
                                            tf.reduce_min(self.kl_scale)),
                          tf.summary.scalar("kl_scaling_dist_max",
                                            tf.reduce_max(self.kl_scale))]

        # Merge all the summaries, but ensure we are post-activation
        with tf.control_dependencies([self.x_reconstr_mean_activ]):
            self.summaries = tf.summary.merge(summaries)

        # Write all summaries to logs, but VARY the model name AND add a TIMESTAMP
        current_summary_name = self.get_name() + self.get_formatted_datetime()
        self.summary_writer = tf.summary.FileWriter("logs/" + current_summary_name,
                                                    self.sess.graph, flush_secs=60)


    '''
    A helper function to format the name as a function of the hyper-parameters
    '''
    def get_name(self):
        if self.submodel == 0:
            full_hash_str = self.activation.__name__ + '_' \
                            + str(self.encoder_model.get_sizing()) \
                            + str(self.decoder_model.get_sizing()) \
                            + "_learningrate" + str(self.learning_rate) \
                            + "_latent size" + str(self.latent_size)
            full_hash_str = full_hash_str.strip().lower().replace('[', '')  \
                                                         .replace(']', '')  \
                                                         .replace(' ', '')  \
                                                         .replace('{', '') \
                                                         .replace('}', '') \
                                                         .replace(',', '_') \
                                                         .replace(':', '') \
                                                         .replace('\'', '')
            return 'vae%d_' % self.submodel + full_hash_str
        else:
            vae_tm1_name = self.vae_tm1.get_name()
            indexof = vae_tm1_name.find('_')
            return 'vae%d_' % self.submodel + vae_tm1_name[indexof+1:]

    def get_formatted_datetime(self):
        return str(datetime.datetime.now()).replace(" ", "_") \
                                           .replace("-", "_") \
                                           .replace(":", "_")

    def save(self):
        model_filename = "models/%s.cpkt" % self.get_name()
        print 'saving vae model to %s...' % model_filename
        self.saver.save(self.sess, model_filename)

    def restore(self):
        model_filename = "models/%s.cpkt" % self.get_name()
        print 'into restore, model name = ', model_filename
        if os.path.isfile(model_filename):
            print 'restoring vae model from %s...' % model_filename
            self.saver.restore(self.sess, model_filename)

    @staticmethod
    def kl_categorical(p=None, q=None, p_logits=None, q_logits=None, eps=1e-6):
        '''
        Given p and q (as EITHER BOTH logits or softmax's)
        then this func returns the KL between them.

        Utilizes an eps in order to resolve divide by zero / log issues
        '''
        if p_logits is not None and q_logits is not None:
            Q = distributions.Categorical(logits=q_logits, dtype=tf.float32)
            P = distributions.Categorical(logits=p_logits, dtype=tf.float32)
        elif p is not None and q is not None:
            print 'p shp = ', p.get_shape().as_list(), \
                ' | q shp = ', q.get_shape().as_list()
            Q = distributions.Categorical(p=q+eps, dtype=tf.float32)
            P = distributions.Categorical(p=p+eps, dtype=tf.float32)
        else:
            raise Exception("please provide either logits or dists")

        return distributions.kl(P, Q)

    @staticmethod
    def zero_pad_smaller_cat(cat1, cat2):
        c1shp = cat1.get_shape().as_list()
        c2shp = cat2.get_shape().as_list()
        diff = abs(c1shp[1] - c2shp[1])

        # blend in extra zeros appropriately
        if c1shp[1] > c2shp[1]:
            cat2 = tf.concat([cat2, tf.zeros([c2shp[0], diff])], axis=1)
        elif c2shp[1] > c1shp[1]:
            cat1 = tf.concat([cat1, tf.zeros([c1shp[0], diff])], axis=1)

        return [cat1, cat2]

    def _create_constraints(self):
        # 0.) add in a kl term between the old model's posterior
        #     and the current model's posterior using the
        #     data generated from the previous model [for the discrete ONLY]
        #
        # Recall data is : [current_data ; old_data]
        if self.submodel > 0:
            # First we encode the generated data w/the student
            # Note: encode returns z, z_normal, z_discrete,
            #                      kl_normal, kl_discrete
            # Note2: discrete dimension is self.submodel
            self.q_z_s_given_x_t, _, _, _ \
                = self.encoder(self.xhat_tm1,
                               rnd_sample=None,
                               hard=False,  # True?
                               reuse=True)

            # We also need to encode the data back through the teacher
            # This is necessary because we need to evaluate the posterior
            # in order to compare Q^T(x|z) against Q^S(x|z)
            # Note2: discrete dimension is self.submodel - 1 [possibly?]
            self.q_z_t_given_x_t, _, _, _ \
                = self.vae_tm1.encoder(self.xhat_tm1,
                                       rnd_sample=None,
                                       hard=False,  # True?
                                       reuse=True)

            # Get the number of gaussians for student and teacher
            # We also only consider num_old_data of the batch
            qzt_shp = self.q_z_t_given_x_t.get_shape().as_list()
            qzs_shp = self.q_z_s_given_x_t.get_shape().as_list()
            ng_t = qzt_shp[0] - self.vae_tm1.num_discrete  # num gaussians(t)
            ng_s = qzs_shp[0] - self.num_discrete          # num gaussians(s)
            self.q_z_s_given_x_t = self.q_z_s_given_x_t[0:self.num_old_data, ng_s:]
            self.q_z_t_given_x_t = self.q_z_t_given_x_t[0:self.num_old_data, ng_t:]
            self.q_z_s_given_x_t, self.q_z_t_given_x_t \
                = VAE.zero_pad_smaller_cat(self.q_z_s_given_x_t,
                                           self.q_z_t_given_x_t)

            # Now we ONLY want eval the KL on the discrete z
            kl = self.kl_categorical(q=self.q_z_t_given_x_t,
                                     p=self.q_z_s_given_x_t)
            print 'kl_consistency [prepad] : ', kl.get_shape().as_list()
            kl = [tf.zeros([self.num_current_data]), kl]
            self.kl_consistency = tf.concat(axis=0, values=kl)
        else:
            self.q_z_given_x = tf.zeros_like(self.x)
            self.kl_consistency = tf.zeros([self.batch_size], dtype=tf.float32)

    @staticmethod
    def reparameterize(encoded, num_discrete, tau, hard=False,
                       rnd_sample=None, eps=1e-20):
        eshp = encoded.get_shape().as_list()
        print 'esp = ', eshp
        num_normal = eshp[1] - num_discrete
        print 'num_normal = ', num_normal
        logits_normal = encoded[:, 0:num_normal]
        logits_gumbel = encoded[:, num_normal:eshp[1]]

        # we reparameterize using both the N(0, I) and the gumbel(0, 1)
        z_discrete, kl_discrete = gumbel_reparmeterization(logits_gumbel,
                                                           tau,
                                                           rnd_sample)
        z_n, kl_n = gaussian_reparmeterization(logits_normal)

        # merge and pad appropriately
        kl_discrete = tf.concat([tf.zeros(num_normal), kl_discrete], axis=0)
        kl_n = tf.concat([kl_n, tf.zeros(num_discrete)], axis=0)
        z = tf.concat([z_n, z_discrete], axis=1)

        return [slim.flatten(z),
                slim.flatten(z_n),
                slim.flatten(z_discrete),
                kl_n,
                kl_discrete]

    def encoder(self, X, rnd_sample=None, reuse=False, hard=False):
        with tf.variable_scope(self.get_name() + "/encoder", reuse=reuse):
            encoded = forward(X, self.encoder_model)
            return VAE.reparameterize(encoded, self.num_discrete,
                                      self.tau, hard=hard,
                                      rnd_sample=rnd_sample)

    def generator(self, Z, reuse=False):
        with tf.variable_scope(self.get_name() + "/generator", reuse=reuse):
            print 'generator scope: ', tf.get_variable_scope().name
            # Use generator to determine mean of
            # Bernoulli distribution of reconstructed input
            # print 'batch norm for decoder: ', use_ln
            return forward(Z, self.decoder_model)

    def _augment_data(self):
        '''
        Augments [current_data ; old_data]
        '''
        def _train():
            if hasattr(self, 'xhat_tm1'):  # make sure we have forked
                # zero pad the current data on the bottom and add to
                # the data we generated in _generate_vae_tm1_data()
                full_data = [self.x[0:self.num_current_data],
                             self.xhat_tm1[0:self.num_old_data]]
                combined = tf.concat(axis=0, values=full_data,
                                     name="current_data")
            else:
                combined = self.x

            print 'augmented data = ', combined.get_shape().as_list()
            return combined

        def _test():
            return self.x

        return tf.cond(self.is_training, _train, _test)

    def _generate_vae_tm1_data(self):
        if self.vae_tm1 is not None:
            num_instances = self.x.get_shape().as_list()[0]
            self.num_current_data = int((1.0/(self.submodel + 1.0))
                                        * float(num_instances))
            self.num_old_data = num_instances - self.num_current_data
            # TODO: Remove debug trace
            print 'total instances: %d | current_model: %d | current data number: %d | old data number: %d' \
                % (num_instances, self.submodel, self.num_current_data, self.num_old_data)

            if self.num_old_data > 0:  # make sure we aren't in base case
                # generate data by randomly sampling a categorical for
                # N-1 positions; also sample a N(0, I) in order to
                # generate variability
                self.z_tm1, self.xhat_tm1 \
                    = self.generate_at_least(self.vae_tm1,
                                             self.batch_size)

                print 'z_tm1 = ', self.z_tm1.get_shape().as_list(), \
                    '| xhat_tm1 = ', self.xhat_tm1.get_shape().as_list()

    @staticmethod
    def _z_to_one_hot(z, latent_size):
        indices = tf.arg_max(z, 1)
        return tf.one_hot(indices, latent_size, dtype=tf.float32)

    def _shuffle_all_data_together(self):
        if not hasattr(self, 'shuffle_indices'):
            self.shuffle_indices = np.random.permutation(self.batch_size)

        if self.vae_tm1 is not None:
            # we get the total size of the cols and jointly shuffle
            # using the perms generated above.
            self.x_augmented = shuffle_rows_based_on_indices(self.shuffle_indices,
                                                             self.x_augmented)

    '''
    Helper op to create the network structure
    '''
    def _create_network(self, num_test_memories=10):
        self.num_current_data = self.x.get_shape().as_list()[0]

        # generate & shuffle data together
        self._generate_vae_tm1_data()
        self.x_augmented = self._augment_data()
        assert self.x_augmented.get_shape().as_list() \
            == self.x.get_shape().as_list()
        print 'xaug = ', self.x_augmented.get_shape().as_list()
        # TODO: self._shuffle_all_data_together() possible?

        # run the encoder operation
        self.z, \
            self.z_normal,\
            self.z_discrete, \
            self.kl_normal, \
            self.kl_discrete = self.encoder(self.x_augmented,
                                            rnd_sample=None)
        print 'z_encoded = ', self.z.get_shape().as_list()

        # reconstruct x via the generator & run activation
        self.x_reconstr_mean = self.generator(self.z)
        self.x_reconstr_mean_activ = tf.nn.sigmoid(self.x_reconstr_mean)
        # self.x_reconstr = distributions.Bernoulli(logits=self.x_reconstr_logits)
        # self.x_reconstr_mean_activ = self.x_reconstr.mean()

    def _loss_helper(self, truth, pred):
        if self.target_dataset == "mnist":
            loss = self._cross_entropy(truth, pred)
        else:
            loss = self._l2_loss(truth, pred)

        return tf.reduce_sum(loss, 1)

    @staticmethod
    def _cross_entropy(x, x_reconstr):
        # To ensure stability and avoid overflow, the implementation uses
        # max(x, 0) - x * z + log(1 + exp(-abs(x)))
        # return tf.maximum(x, 0) - x * z + tf.log(1.0 + tf.exp(-tf.abs(x)))
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_reconstr,
                                                       labels=x)

    @staticmethod
    def _l2_loss(x, x_reconstr):
        return tf.square(x - x_reconstr)

    def vae_loss(self, x, x_reconstr_mean, latent_kl, consistency_kl):
        # the loss is composed of two terms:
        # 1.) the reconstruction loss (the negative log probability
        #     of the input under the reconstructed bernoulli distribution
        #     induced by the decoder in the data space).
        #     this can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # reconstr_loss = tf.reduce_sum(x_reconstr_mean.log_pmf(x), [1])
        reconstr_loss = self._loss_helper(x, x_reconstr_mean)

        # 2.) the latent loss, which is defined as the kullback leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. this acts as a kind of regularizer.
        #     this can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        # kl_categorical(p=none, q=none, p_logits=none, q_logits=none, eps=1e-6):
        # cost = reconstr_loss - latent_kl
        cost = reconstr_loss + latent_kl + consistency_kl

        # create the reductions only once
        latent_loss_mean = tf.reduce_mean(latent_kl)
        reconstr_loss_mean = tf.reduce_mean(reconstr_loss)
        cost_mean = tf.reduce_mean(cost)

        return [reconstr_loss, reconstr_loss_mean,
                latent_loss_mean, cost, cost_mean]

    def generate_at_least(self, vae_tm1, batch_size):
        # Returns :
        # 1) a categorical and a Normal distribution concatenated
        # 2) x_hat_tm1 : the reconstructed data from the old model
        z_cat = generate_random_categorical(vae_tm1.num_discrete,
                                            batch_size)
        z_normal = tf.random_normal([batch_size, vae_tm1.latent_size])
        z = tf.concat([z_normal, z_cat])
        zshp = z.get_shape().as_list()  # TODO: debug trace
        print 'z internal shp = ', zshp

        # Generate reconstructions of historical Z's
        xr = tf.stop_gradient(tf.nn.sigmoid(vae_tm1.generator(z, reuse=True)))
        print 'xhat internal shp = ', xr.get_shape().as_list()  # TODO: debug

        return z, xr

    def _create_loss_optimizer(self):
        # build constraint graph
        self._create_constraints()

        with tf.variable_scope(self.get_name() + "/loss_optimizer"):
            # set the indexes of the latent_kl to zero for the
            # indices that we are constraining over as we are computing
            # a regularizer in the above function
            if self.submodel > 0:
                zero_vals = [self.latent_kl[0:self.num_current_data],
                             tf.zeros([self.num_old_data])]
                self.latent_kl = tf.concat(axis=0, values=zero_vals)

            # tabulate total loss
            self.reconstr_loss, self.reconstr_loss_mean, \
                self.latent_loss_mean, \
                self.cost, self.cost_mean \
                = self.vae_loss(self.x_augmented,
                                self.x_reconstr_mean,
                                self.kl_normal + self.kl_discrete,
                                self.kl_consistency)

            # construct our optimizer
            with tf.control_dependencies([self.x_reconstr_mean_activ]):
                filtered = [v for v in tf.trainable_variables()
                            if v.name.startswith(self.get_name())]
                self.optimizer = self._create_optimizer(filtered,
                                                        self.cost_mean,
                                                        self.learning_rate)

    def _create_optimizer(self, tvars, cost, lr):
        # optimizer = tf.train.rmspropoptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        print 'there are %d trainable vars in cost %s\n' % (len(tvars), cost.name)
        # grads = tf.gradients(cost, tvars)

        # DEBUG: exploding gradients test with this:
        # for index in range(len(grads)):
        #     if grads[index] is not None:
        #         gradstr = "\n grad [%i] | tvar [%s] =" % (index, tvars[index].name)
        #         grads[index] = tf.Print(grads[index], [grads[index]], gradstr, summarize=100)

        # grads, _ = tf.clip_by_global_norm(grads, 5.0)
        # return optimizer.apply_gradients(zip(grads, tvars))
        return tf.train.AdamOptimizer(learning_rate=lr).minimize(cost, var_list=tvars)

    def partial_fit(self, inputs, iteration_print=10, is_forked=False):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """

        feed_dict = {self.x: inputs,
                     self.is_training: True,
                     self.tau: self.tau_host}

        try:
            if self.iteration > 0 and self.iteration % 10 == 0:
                rate = -self.anneal_rate*self.iteration
                self.tau_host = np.maximum(self.tau0 * np.exp(rate),
                                           self.min_temp)
                print 'updated tau to ', self.tau_host

            ops_to_run = [self.optimizer, self.iteration_gpu_op,
                          self.kl_consistency, self.latent_kl, self.q_z_given_x,
                          self.z, self.cost_mean, self.reconstr_loss_mean,
                          self.latent_loss_mean]

            if self.iteration % iteration_print == 0:
                _, _, kld, lkl, zhat_t, z, \
                    cost, rloss, lloss, summary \
                    = self.sess.run(ops_to_run + [self.summaries],
                                    feed_dict=feed_dict)

                self.summary_writer.add_summary(summary, self.iteration
                                                * iteration_print)
            else:
                _, _, kld, lkl, zhat_t, z, \
                    cost, rloss, lloss \
                    = self.sess.run(ops_to_run,
                                    feed_dict=feed_dict)

        except Exception as e:
            print 'caught exception in partial fit: ', e

        if hasattr(self, 'xhat_tm1'):
            print 'zhat_t = ', np.argmax(zhat_t, axis=1), ' | ',

        print "latent_kl = ", np.sum(lkl), " | kl_distill = ", np.sum(kld)

        self.iteration += 1
        return cost, rloss, lloss

    def write_classes_to_file(self, filename, all_classes):
        with open(filename, 'a') as f:
            np.savetxt(f, self.sess.run(all_classes), delimiter=",")

    def fork(self):
        '''
        Fork the current model by copying the model parameters
        into the old ones.

        Note: This is a slow op in tensorflow
              because the session needs to be run
        '''
        updated_latent_size = self.latent_size + 1  # XXX: compute this
        encoder = DenseEncoder(self.sess, updated_latent_size,
                               self.is_training,
                               use_ln=self.encoder_model.use_ln,
                               use_bn=self.decoder_model.use_bn,
                               activate_last_layer=False)
        decoder = DenseEncoder(self.sess, self.input_size,
                               self.is_training,
                               use_ln=self.decoder_model.use_ln,
                               use_bn=self.decoder_model.use_bn,
                               activate_last_layer=False)

        vae_tp1 = VAE(self.sess, self.input_size, self.batch_size,
                      latent_size=self.latent_size,
                      encoder=encoder,
                      decoder=decoder,
                      activation=self.activation,
                      learning_rate=self.learning_rate,
                      submodel=self.submodel+1,
                      vae_tm1=self)

        # we want to reinit our weights and biases to their defaults
        # TODO: Evaluate whether simply copying over weights will be better
        self.sess.run(vae_tp1.init_op)
        return vae_tp1

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z, feed_dict={self.x: X,
                                                self.tau: self.tau_host,
                                                self.is_training: False,
                                                self.dropout_keep_prob: 1.0})

    def generate(self, z=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z is None:
            z = generate_random_categorical(self.latent_size, self.batch_size)

        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean_activ,
                             feed_dict={self.z: z,
                                        self.tau: self.tau_host,
                                        self.is_training: False})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean_activ,
                             feed_dict={self.x: X,
                                        self.tau: self.tau_host,
                                        self.is_training: False})

    def train(self, source, batch_size, training_epochs=10, display_step=5):
        n_samples = source.train.num_examples
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, _ = source.train.next_batch(batch_size)

                # Fit training using batch data
                cost, recon_cost, latent_cost = self.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print "[Epoch:", '%04d]' % (epoch+1), \
                    "current cost = ", "{:.4f} | ".format(cost), \
                    "avg cost = ", "{:.4f} | ".format(avg_cost), \
                    "latent cost = ", "{:.4f} | ".format(latent_cost), \
                    "recon cost = ", "{:.4f}".format(recon_cost)

def build_Nd_vae(sess, source, input_shape, latent_size, batch_size, epochs=100):
    latest_model = find_latest_file("models", "vae(\d+)")
    print 'latest model = ', latest_model

    # build encoder and decoder models
    # note: these can be externally built
    #       as long as it works with forward()
    is_training = tf.placeholder(tf.bool)
    encoder = DenseEncoder(sess, 2*FLAGS.latent_size + 1,
                           is_training,
                           use_ln=FLAGS.use_ln,
                           use_bn=FLAGS.use_bn,
                           activate_last_layer=False)
    decoder = DenseEncoder(sess, input_shape,
                           is_training,
                           use_ln=FLAGS.use_ln,
                           use_bn=FLAGS.use_bn,
                           activate_last_layer=False)
    print 'encoder = ', encoder.get_info()
    print 'decoder = ', decoder.get_info()

    # build the vae object
    vae = VAE(sess, input_shape, FLAGS.batch_size,
              latent_size=FLAGS.latent_size,
              encoder=encoder, decoder=decoder,
              learning_rate=FLAGS.learning_rate,
              submodel=latest_model[1],
              vae_tm1=None)

    model_filename = "models/%s" % latest_model[0]
    is_forked = False

    if os.path.isfile(model_filename):
        vae.restore()
    else:
        # initialize all the variables
        sess.run(tf.global_variables_initializer())

        try:
            if FLAGS.target_dataset == "mnist" and not FLAGS.sequential:
                vae.train(source[0], batch_size, display_step=1, training_epochs=epochs)
            else:
                current_model = 0
                for epoch in range(int(1e6)):
                    # fork if we get a new model
                    prev_model = current_model
                    inputs, outputs, indexes, current_model = generate_train_data(source,
                                                                                  batch_size,
                                                                                  batch_size,
                                                                                  current_model)
                    if prev_model != current_model:
                        previous_data, _, _ = _generate_from_index(source, [prev_model]*batch_size)
                        # plt.figure()
                        # plt.imshow(previous_data[0].reshape(28, 28))
                        # plt.savefig("imgs/x_tm1.png", bbox_inches='tight')
                        vae = vae.fork()
                        is_forked = True

                    for start, end in zip(range(0, len(inputs) + 1, batch_size),
                                          range(batch_size, len(inputs) + 1, batch_size)):
                        #x = np.hstack([inputs[start:end], outputs[start:end]])
                        x = outputs[start:end] if FLAGS.target_dataset == "regression" else inputs[start:end]
                        loss, rloss, lloss = vae.partial_fit(x, is_forked=is_forked)
                        print 'loss[total_iter=%d][iter=%d][model=%d] = %f, latent loss = %f, reconstr loss = %f' \
                            % (epoch, vae.iteration, current_model, loss, lloss,
                               rloss if rloss is not None else 0.0)

        except KeyboardInterrupt:
            print "caught keyboard exception..."

        vae.save()

    return vae

# show clustering in 2d
def plot_2d_vae(sess, x_sample, y_sample, vae, batch_size):
    x_sample = np.asarray(x_sample)
    y_sample = np.asarray(y_sample)
    print 'xs = ', x_sample.shape, ' | ys = ', y_sample.shape

    z_mu = []
    for start, end in zip(range(0, y_sample.shape[0] + 1, batch_size), \
                          range(batch_size, y_sample.shape[0] + 1, batch_size)):
        z_mu.append(vae.transform(x_sample[start:end]))

    z_mu = np.vstack(z_mu)
    # z_mu, c = reject_outliers(np.vstack(z_mu), np.argmax(y_sample, 1))
    # print 'zmus = ', z_mu.shape, ' c = ', c.shape

    plt.figure(figsize=(8, 6))

    # plt.ylim(-0.25, 0.25)
    # plt.xlim(-0.25, 0.25)

    if FLAGS.target_dataset == "mnist":
        #plt.scatter(z_mu[:, 0], z_mu[:, 1], c=c) # for reject_outliers
        c = np.argmax(y_sample, 1) if len(y_sample.shape) > 1 else y_sample
        plt.scatter(z_mu[:, 0], z_mu[:, 1], c=c)
    elif FLAGS.target_dataset == "regression":
        plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y_sample)
        plt.colorbar()
        plt.savefig("imgs/2d_cluster_orig_%s.png" % vae.get_name(),
                    bbox_inches='tight')

    plt.colorbar()
    plt.savefig("imgs/2d_cluster_%s.png" % vae.get_name(),
                bbox_inches='tight')
    plt.show()

def _write_images(x_sample, x_reconstruct, vae_name, filename=None, num_print=5, sup_title=None):
    fig = plt.figure(figsize=(8, 12))
    if sup_title:
        fig.suptitle(sup_title)

    for i in range(num_print):
        if x_sample is not None:
            plt.subplot(num_print, 2, 2*i + 1)
            plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
            plt.title("Test input")
            plt.colorbar()

        plt.subplot(num_print, 2, 2*i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1)
        plt.title("Reconstruction")
        plt.colorbar()

    if filename is None:
        plt.savefig("imgs/20d_reconstr_%d_%s.png" % (i, vae_name), bbox_inches='tight')
    else:
        plt.savefig(filename, bbox_inches='tight')

    plt.close()


def plot_ND_vae_inference(sess, vae, batch_size, num_write=10):
    z_generated = generate_random_categorical(FLAGS.latent_size, batch_size)
    vae_i = vae; current_vae = 0
    while vae_i is not None: # do this for all the forked VAE's
        x_reconstruct = vae_i.generate(z_mu=z_generated)
        for x,z in zip(x_reconstruct[0:num_write], z_generated[0:num_write]): # only write num_write images
            #current_pred_str = '_'.join(map(str, index_of_generation))
            current_pred_str = '_atindex' + str(np.argwhere(z)[0][0])
            plt.figure()
            plt.title(current_pred_str)
            plt.imshow(x.reshape(28, 28), vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig("imgs/vae_%d_inference_%s.png" % (current_vae, current_pred_str),
                        bbox_inches='tight')
            plt.close()
            print 'z_generated[vae# %d] = %s' % (current_vae, current_pred_str)

        vae_i = vae_i.vae_tm1
        current_vae += 1

def find_root_vae(vae_i):
    root_vae = vae_i
    while root_vae and root_vae.vae_tm1:
        root_vae = root_vae.vae_tm1

    return root_vae

# plot the evaluation of all the models on the provided running Z logits
def plot_vae_consistency(sess, vae, z_test, z_from, batch_size, num_write=5):
    vae_i = vae

    while vae_i is not None: # do this for all the forked VAE's
        x_reconstruct = vae_i.generate(z=z_test)
        consistency_str = '%d_consistency_using_zavg_from_%s' % (vae_i.submodel, z_from)
        _write_images(x_sample=None,
                      x_reconstruct=x_reconstruct,
                      vae_name=None,
                      filename="imgs/vae_%s.png" % consistency_str,
                      num_print=num_write,
                      sup_title=consistency_str)

        # for x in x_reconstruct[0:num_write]: # only write num_write images
        #     consistency_str = '%d_consistency_using_zavg_from_%s' % (vae_i.submodel, z_from)

        #     plt.figure()
        #     plt.title(consistency_str)
        #     plt.imshow(x.reshape(28, 28), vmin=0, vmax=1)
        #     plt.colorbar()
        #     plt.savefig("imgs/consistency/vae_%s.png" % (consistency_str),
        #                 bbox_inches='tight')
        #     plt.close()

        vae_i = vae_i.vae_tm1

    # show reconstruction
def plot_Nd_vae(sess, source, vae, batch_size):
    if FLAGS.target_dataset == "mnist" and not FLAGS.sequential:
        x_sample = source[0].test.next_batch(batch_size)[0]
        x_reconstruct = vae.reconstruct(x_sample)
    elif FLAGS.target_dataset == "mnist" and FLAGS.sequential:
        from tensorflow.examples.tutorials.mnist import input_data
        x_sample = input_data.read_data_sets('MNIST_data', one_hot=True).test.next_batch(batch_size)[0]
        x_reconstruct = vae.reconstruct(x_sample)
        x_reconstruct_tm1 = []
        vae_tm1 = vae.vae_tm1
        while vae_tm1 is not None:
            x_reconstruct_tm1.append([vae_tm1.reconstruct(x_sample), vae_tm1.get_name()])
            vae_tm1 = vae_tm1.vae_tm1
    else:
        x_sample, y_sample, indexes = generate_test_data(source, batch_size, FLAGS.batch_size)
        x = y_sample
        x_reconstruct = vae.reconstruct(x)

    _write_images(x_sample, x_reconstruct, vae.get_name())
    for x_r_tm1, name_tm1 in x_reconstruct_tm1:
        _write_images(x_sample, x_r_tm1, name_tm1)

def create_indexes(num_train, num_models, current_model):
    global TRAIN_ITER
    global GLOBAL_ITER
    if np.random.randint(0, FLAGS.batch_size * 13) == 2 and TRAIN_ITER > FLAGS.min_interval: # XXX: const 5k
        #current_model = np.random.randint(0, num_models)
        current_model += 1 if current_model < num_models - 1 else 0
        TRAIN_ITER = 0

    GLOBAL_ITER += 1
    TRAIN_ITER += 1

    return current_model, [current_model] * num_train

def _generate_from_index(generators, gen_indexes):
    try:
        full_data = [generators[t].get_batch_iter(1) for t in gen_indexes]
        inputs = np.vstack([t[0] for t in full_data])
        outputs = np.vstack([t[1] for t in full_data])
        return inputs, outputs, gen_indexes
    except Exception as e:
        print 'caught exception in gen_from_index: ', e
        print 'len generators = %d | t = %d' % (len(generators), t)


def generate_train_data(generators, num_train, batch_size, current_model):
    current_model, indexes = create_indexes(num_train, len(generators), current_model)
    num_batches = int(np.floor(len(indexes) / batch_size))
    indexes = indexes[0:num_batches * batch_size] # dump extra data
    inputs, outputs, _ = _generate_from_index(generators, indexes)
    return inputs, outputs, indexes, current_model

def generate_test_data(generators, num_train, batch_size):
    indexes = list(np.arange(len(generators))) * num_train
    num_batches = int(np.floor(len(indexes) / batch_size))
    indexes = indexes[0:num_batches * batch_size] # dump extra data
    return _generate_from_index(generators, indexes)

def evaluate_running_hist(vae):
    vae_t = vae
    current_vae = 0
    while vae_t is not None:
        print 'histogram[vae# %d]' % current_vae, vae_t.running_hist_host
        vae_t = vae_t.vae_tm1
        current_vae += 1

def main():
    if FLAGS.target_dataset == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        generators = [MNIST_Number(i, full_mnist, False) for i in xrange(10)] if FLAGS.sequential  \
                     else [input_data.read_data_sets('MNIST_data', one_hot=True)]

        input_shape = full_mnist.train.images.shape[1]
    elif FLAGS.target_dataset == "regression":
        #input_shape = 128 * 2
        input_shape = 128
        generators = [RegressionGenerator(w, 128, sequential=False) for w in regression_keys]

    # model storage
    if not os.path.exists('models'):
        os.makedirs('models')

    # img storage
    if not os.path.exists('imgs'):
        os.makedirs('imgs')

    with tf.device(FLAGS.device):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.device_percentage)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                              gpu_options=gpu_options)) as sess:
            vae = build_Nd_vae(sess, generators,
                               input_shape,
                               FLAGS.latent_size,
                               FLAGS.batch_size,
                               epochs=FLAGS.epochs)

            # run a test inference and verify
            if FLAGS.target_dataset == "mnist" and FLAGS.sequential:
                for g, i in zip(generators, range(len(generators))):
                    x_sample, y_sample = g.get_test_batch_iter(FLAGS.batch_size)
                    latent_projection = vae.transform(x_sample)
                    print '############### Testing inference on class %d ##################' % i
                    print 'full latent projection = ', latent_projection.shape
                    #print 'predicted[%d] = ' % i, latent_projection[0]
                    #hist, edges = np.histogram(latent_projection, bins=FLAGS.latent_size, range=[0.0, 1.0])
                    #print 'test hist[%d] = ' % i, hist
                    #print 'test hist edges[%d] = ' % i, edges
                    print 'test hist_argmax[%d] = ' % i, np.argmax(latent_projection, axis=1)
                    print '################################################################'

                # We dont need a generator to test inference or the histograms
                print '\n############### Evaluating Model Consistencies #####################'
                vae_consistency = vae
                while vae_consistency is not None:
                    z_test_i = vae_consistency.running_Z_logits_host
                    plot_vae_consistency(sess, vae, z_test_i,
                                         vae_consistency.submodel,
                                         FLAGS.batch_size)
                    vae_consistency = vae_consistency.vae_tm1

                print '.......done [see imgs/vae_consistency_*]'
                print '#################################################################'

                # print '\n############### Testing generation on [%d] #####################' % vae.submodel
                # plot_ND_vae_inference(sess, vae, FLAGS.batch_size)
                # print '#################################################################'



                # print '\n############### Histogram Distr on Class %d ###################' % i
                # print evaluate_running_hist(vae)
                # print '###############################################################'

            else:
                for i in range(100):
                    x_sample, y_sample = generators[0].test.next_batch(FLAGS.batch_size)
                    latent_projection = vae.transform(x_sample)
                    print 'full latent projection = ', latent_projection.shape
                    print 'predicted[%d][class = %s] = ' % (i, str(y_sample[0])), latent_projection[0]


            # 2d plot shows a cluster plot vs. a reconstruction plot
            if FLAGS.latent_size == 2:
                if FLAGS.target_dataset == "mnist" and not FLAGS.sequential:
                    x_sample, y_sample = generators[0].test.next_batch(10000)
                    plot_2d_vae(sess, x_sample, y_sample, vae, FLAGS.batch_size)
                elif FLAGS.target_dataset == "mnist" and FLAGS.sequential:
                    #x_sample, y_sample = input_data.read_data_sets('MNIST_data', one_hot=True).test.next_batch(10000)
                    x_sample, y_sample = generators[0].get_test_batch_iter(1000)
                    plot_2d_vae(sess, x_sample, y_sample, vae, FLAGS.batch_size)
                elif FLAGS.target_dataset == "regression":
                    # [inputs, outputs, indexes]
                    x_sample, y_sample, indexes = generate_test_data(generators, 10000, FLAGS.batch_size)
                    plot_2d_vae(sess, y_sample, indexes, vae, FLAGS.batch_size)
            else:
                plot_Nd_vae(sess, generators, vae, FLAGS.batch_size)

if __name__ == "__main__":
    main()
