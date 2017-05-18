import os, sys
sys.path.append(os.getcwd())

import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d


class WGAN_GP(object):
    def __init__(self, x, batch_size, img_shape=[28, 28, 1],
                 lambda_reg=10, critics=5, filter_dim=64):
        self.unrolled_dim = np.prod(img_shape)
        self.batch_size = batch_size
        self.critics = critics
        self.lambda_reg = lambda_reg
        self.filter_dim = filter_dim

        unrolled = tf.reshape(x, [-1, self.unrolled_dim])
        real_data = 2*(unrolled - .5)
        fake_data = WGAN_GP.Generator(BATCH_SIZE)

        disc_real = WGAN_GP.Discriminator(real_data)
        disc_fake = WGAN_GP.Discriminator(fake_data)

        gen_params = WGAN_GP.params_with_name('Generator')
        disc_params = WGAN_GP.params_with_name('Discriminator')

        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Gradient penalty
        alpha = tf.random_uniform(
            shape=[batch_size, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        gradients = tf.gradients(WGAN_GP.Discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        disc_cost += lambda_reg*gradient_penalty

        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

        # For generating samples
        fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
        fixed_noise_samples_128 = self.Generator(128, noise=fixed_noise_128)

    def partial_fit(self, inputs, iteration_print=10,
                    iteration_save_imgs=2000,
                    is_forked=False):
        # Train critic
        for i in xrange(self.critics):
            _data = gen.next()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data})

                lib.plot.plot('train disc cost', _disc_cost)
                lib.plot.plot('time', time.time() - start_time)

                # Calculate inception score every 1K iters
                if iteration % 1000 == 999:
                    inception_score = get_inception_score()
                    lib.plot.plot('inception score', inception_score[0])

                # Calculate dev loss and generate samples every 100 iters
                if iteration % 100 == 99:
                    dev_disc_costs = []
                    for images in dev_gen():
                        _dev_disc_cost = session.run(disc_cost, feed_dict={real_data_int: images})
                        dev_disc_costs.append(_dev_disc_cost)
                    lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                    generate_image(iteration, _data)

                # Save logs every 100 iters
                if (iteration < 5) or (iteration % 100 == 99):
                    lib.plot.flush()

                lib.plot.tick()

    @staticmethod
    def params_with_name(name):
        return [p for n,p in _params.items() if name in n]

    @staticmethod
    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)

    @staticmethod
    def ReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
        return tf.nn.relu(output)

    @staticmethod
    def LeakyReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
        return WGAN_GP.LeakyReLU(output)

    @staticmethod
    def Generator(n_samples, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*DIM, 4, 4])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 3, 5, output)

        output = tf.tanh(output)

        return tf.reshape(output, [-1, OUTPUT_DIM])

    @staticmethod
    def Discriminator(inputs):
        output = tf.reshape(inputs, [-1, 3, 32, 32])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
        output = WGAN_GP.LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
        if MODE != 'wgan-gp':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
        output = WGAN_GP.LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
        if MODE != 'wgan-gp':
            output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
        output = WGAN_GP.LeakyReLU(output)

        output = tf.reshape(output, [-1, 4*4*4*DIM])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

        return tf.reshape(output, [-1])
