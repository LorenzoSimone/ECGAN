import tensorflow as tf
import numpy as np
import data_utils
import pdb
import json
tf.logging.set_verbosity(tf.logging.ERROR)
import mmd

# --- Latent Space Sampling Functions --- #

def sample_Z(batch_size, seq_length, latent_dim, use_time=False, use_noisy_time=False):
    """
    Generate random latent space samples based on Gaussian distribution.
    Optionally includes time-based manipulation.
    """
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))  # Gaussian noise
    if use_time:
        print('WARNING: use_time has different semantics')
        sample[:, :, 0] = np.linspace(0, 1.0 / seq_length, num=seq_length)  # Linearly spaced time
    return sample

def sample_C(batch_size, cond_dim=0, max_val=1, one_hot=False):
    """
    Generate conditional samples, either random or one-hot encoded labels.
    """
    if cond_dim == 0:
        return None  # No conditional dimension
    else:
        if one_hot:
            assert max_val == 1  # Ensure only binary one-hot encoding
            C = np.zeros(shape=(batch_size, cond_dim))
            labels = np.random.choice(cond_dim, batch_size)
            C[np.arange(batch_size), labels] = 1  # One-hot encoding
        else:
            C = np.random.choice(max_val + 1, size=(batch_size, cond_dim))  # Random conditional values
        return C

# --- Training Functions --- #

def train_epoch(epoch, samples, labels, sess, Z, X, CG, CD, CS, D_loss, G_loss, D_solver, G_solver, 
                batch_size, use_time, D_rounds, G_rounds, seq_length, 
                latent_dim, num_generated_features, cond_dim, max_val, WGAN_clip, one_hot):
    """
    Train the generator and discriminator for one epoch.
    Iterates through the dataset, alternating between discriminator and generator updates.
    """
    for batch_idx in range(0, int(len(samples) / batch_size) - (D_rounds + (cond_dim > 0) * G_rounds), D_rounds + (cond_dim > 0) * G_rounds):
        # Update the discriminator
        for d in range(D_rounds):
            X_mb, Y_mb = data_utils.get_batch(samples, batch_size, batch_idx + d, labels)
            Z_mb = sample_Z(batch_size, seq_length, latent_dim, use_time)
            if cond_dim > 0:
                Y_mb = Y_mb.reshape(-1, cond_dim)
                if one_hot:
                    offsets = np.random.choice(cond_dim - 1, batch_size) + 1
                    new_labels = (np.argmax(Y_mb, axis=1) + offsets) % cond_dim
                    Y_wrong = np.zeros_like(Y_mb)
                    Y_wrong[np.arange(batch_size), new_labels] = 1
                else:
                    Y_wrong = 1 - Y_mb
                _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb, CD: Y_mb, CS: Y_wrong, CG: Y_mb})
            else:
                _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb})
            if WGAN_clip:
                _ = sess.run([clip_disc_weights])  # Clip weights if using WGAN
        # Update the generator
        for g in range(G_rounds):
            if cond_dim > 0:
                X_mb, Y_mb = data_utils.get_batch(samples, batch_size, batch_idx + D_rounds + g, labels)
                _ = sess.run(G_solver,
                        feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time), CG: Y_mb})
            else:
                _ = sess.run(G_solver,
                        feed_dict={Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})
    # Calculate and return the current loss
    if cond_dim > 0:
        D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time), CG: Y_mb, CD: Y_mb})
    else:
        D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_mb, Z: sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)})
    D_loss_curr = np.mean(D_loss_curr)
    G_loss_curr = np.mean(G_loss_curr)
    return D_loss_curr, G_loss_curr

def WGAN_loss(Z, X, WGAN_clip=False):
    """
    Define WGAN loss for the discriminator and generator.
    """
    raise NotImplementedError  # Placeholder for unimplemented WGAN loss logic

    # Generate samples and compute discriminator outputs
    G_sample = generator(Z, hidden_units_g, W_out_G, b_out_G, scale_out_G)
    D_real, D_logit_real, D_logit_real_final = discriminator(X, hidden_units_d, seq_length, batch_size)

    # Calculate discriminator and generator loss
    D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real)
    G_loss = -tf.reduce_mean(D_fake)

    if not WGAN_clip:
        # Optionally include gradient penalty
        pass

    return G_loss, D_loss, clip_disc_weights

def GAN_loss(Z, X, generator_settings, discriminator_settings, kappa, cond, CG, CD, CS, wrong_labels=False):
    """
    Define loss functions for GAN training, with optional conditional labels (CGAN).
    """
    if cond:
        G_sample = generator(Z, **generator_settings, c=CG)
        D_real, D_logit_real =  discriminator(X, **discriminator_settings, c=CD)
        D_fake, D_logit_fake = discriminator(G_sample, reuse=True, **discriminator_settings, c=CG)
        if wrong_labels:
            D_wrong, D_logit_wrong = discriminator(X, reuse=True, **discriminator_settings, c=CS)
    else:
        G_sample = generator(Z, **generator_settings)
        D_real, D_logit_real  = discriminator(X, **discriminator_settings)
        D_fake, D_logit_fake = discriminator(G_sample, reuse=True, **discriminator_settings)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)

    D_loss = D_loss_real + D_loss_fake

    if cond and wrong_labels:
        D_loss += D_loss_wrong

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)

    return D_loss, G_loss

def GAN_solvers(D_loss, G_loss, learning_rate, batch_size, total_examples, 
                l2norm_bound, batches_per_lot, sigma, dp=False):
    """
    Define solvers for the GAN training, using standard or differential privacy optimizers.
    """
    discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

    if dp:
        # Differentially private SGD for the discriminator
        eps = tf.placeholder(tf.float32)
        delta = tf.placeholder(tf.float32)
        priv_accountant = accountant.GaussianMomentsAccountant(total_examples)
        gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant, [l2norm_bound, True])
        D_solver = dp_optimizer.DPGradientDescentOptimizer(learning_rate, [eps, delta], sanitizer=gaussian_sanitizer, sigma=sigma).minimize(D_loss, var_list=discriminator_vars)
    else:
        D_loss_mean_over_batch = tf.reduce_mean(D_loss)
        D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss_mean_over_batch, var_list=discriminator_vars)

    G_loss_mean_over_batch = tf.reduce_mean(G_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss_mean_over_batch, var_list=generator_vars)

    return D_solver, G_solver, priv_accountant

# --- Model Functions --- #

def create_placeholders(batch_size, seq_length, latent_dim, num_generated_features, cond_dim):
    """
    Create TensorFlow placeholders for input variables: Z (latent), X (real data), CG, CD, CS (conditional).
    """
    Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    X = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
    CG = tf.placeholder(tf.float32, [batch_size, cond_dim])
    CD = tf.placeholder(tf.float32, [batch_size, cond_dim])
    CS = tf.placeholder(tf.float32, [batch_size, cond_dim])
    return Z, X, CG, CD, CS

def generator(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None, cond_dim=0, c=None, learn_scale=True):
    """
    If parameters are supplied, initialise as such
    """
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_G_initializer = tf.truncated_normal_initializer()
            b_out_G_initializer = tf.truncated_normal_initializer()
            scale_out_G_initializer = tf.constant_initializer(value=1.0)
            lstm_initializer = None
            bias_start = 1.0
        else:
            W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
            b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
            try:
                scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
            except KeyError:
                scale_out_G_initializer = tf.constant_initializer(value=1)
                assert learn_scale
            lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
            bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer, trainable=learn_scale)
        if cond_dim > 0:
            # CGAN!
            assert not c is None
            repeated_encoding = tf.stack([c]*seq_length, axis=1)
            inputs = tf.concat([z, repeated_encoding], axis=2)

            #repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
            #repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1], cond_dim])
            #inputs = tf.concat([repeated_encoding, z], 2)
        else:
            inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                           state_is_tuple=True,
                           initializer=lstm_initializer,
                           bias_start=bias_start,
                           reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length]*batch_size,
            inputs=inputs)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
#        output_2d = tf.multiply(tf.nn.tanh(logits_2d), scale_out_G)
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d

def discriminator(x, hidden_units_d, seq_length, batch_size, reuse=False, 
        cond_dim=0, c=None, batch_mean=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                initializer=tf.truncated_normal_initializer())
        b_out_D = tf.get_variable(name='b_out_D', shape=1,
                initializer=tf.truncated_normal_initializer())
#        W_final_D = tf.get_variable(name='W_final_D', shape=[hidden_units_d, 1],
#                initializer=tf.truncated_normal_initializer())
#        b_final_D = tf.get_variable(name='b_final_D', shape=1,
#                initializer=tf.truncated_normal_initializer())

        if cond_dim > 0:
            assert not c is None
            repeated_encoding = tf.stack([c]*seq_length, axis=1)
            inputs = tf.concat([x, repeated_encoding], axis=2)
        else:
            inputs = x
        # add the average of the inputs to the inputs (mode collapse?
        if batch_mean:
            mean_over_batch = tf.stack([tf.reduce_mean(x, axis=0)]*batch_size, axis=0)
            inputs = tf.concat([x, mean_over_batch], axis=2)

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, 
                state_is_tuple=True,
                reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=inputs)
#        logit_final = tf.matmul(rnn_outputs[:, -1], W_final_D) + b_final_D
        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
#        rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units_d])
#        logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D
        output = tf.nn.sigmoid(logits)
    #return output, logits, logit_final
    return output, logits

# --- to do with saving/loading --- #

def dump_parameters(identifier, sess):
    """
    Save model parmaters to a numpy file
    """
    dump_path = './experiments/parameters/' + identifier + '.npy'
    model_parameters = dict()
    for v in tf.trainable_variables():
        model_parameters[v.name] = sess.run(v)
    np.save(dump_path, model_parameters)
    print('Recorded', len(model_parameters), 'parameters to', dump_path)
    return True

def load_parameters(identifier):
    """
    Load parameters from a numpy file
    """
    load_path = './experiments/parameters/' + identifier + '.npy'
    model_parameters = np.load(load_path).item()
    return model_parameters

# --- to do with trained models --- #

def sample_trained_model(settings, epoch, num_samples, Z_samples=None, C_samples=None):
    """
    Return num_samples samples from a trained model described by settings dict
    """
    # if settings is a string, assume it's an identifier and load
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))
    print('Sampling', num_samples, 'samples from', settings['identifier'], 'at epoch', epoch)
    # get the parameters, get other variables
    parameters = load_parameters(settings['identifier'] + '_' + str(epoch))
    # create placeholder, Z samples
    Z = tf.placeholder(tf.float32, [num_samples, settings['seq_length'], settings['latent_dim']])
    CG = tf.placeholder(tf.float32, [num_samples, settings['cond_dim']])
    if Z_samples is None:
        Z_samples = sample_Z(num_samples, settings['seq_length'], settings['latent_dim'], settings['use_time'], use_noisy_time=False)
    else:
        assert Z_samples.shape[0] == num_samples
    # create the generator (GAN or CGAN)
    if C_samples is None:
        # normal GAN
        G_samples = generator(Z, settings['hidden_units_g'], settings['seq_length'], 
                              num_samples, settings['num_generated_features'], 
                              reuse=False, parameters=parameters, cond_dim=settings['cond_dim'])
    else:
        assert C_samples.shape[0] == num_samples
        # CGAN
        G_samples = generator(Z, settings['hidden_units_g'], settings['seq_length'], 
                              num_samples, settings['num_generated_features'], 
                              reuse=False, parameters=parameters, cond_dim=settings['cond_dim'], c=CG)
    # sample from it 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if C_samples is None:
            real_samples = sess.run(G_samples, feed_dict={Z: Z_samples})
        else:
            real_samples = sess.run(G_samples, feed_dict={Z: Z_samples, CG: C_samples})
    tf.reset_default_graph()
    return real_samples

# --- to do with inversion --- #

def invert(settings, epoch, samples, g_tolerance=None, e_tolerance=0.1,
        n_iter=None, max_iter=10000, heuristic_sigma=None, C_samples=None):
    """
    Return the latent space points corresponding to a set of a samples
    ( from gradient descent )
    """
    # cast samples to float32
    samples = np.float32(samples[:, :, :])
    # get the model
    if type(settings) == str:
        settings = json.load(open('./experiments/settings/' + settings + '.txt', 'r'))
    num_samples = samples.shape[0]
    print('Inverting', num_samples, 'samples using model', settings['identifier'], 'at epoch', epoch,)
    if not g_tolerance is None:
        print('until gradient norm is below', g_tolerance)
    else:
        print('until error is below', e_tolerance)
    # get parameters
    parameters = load_parameters(settings['identifier'] + '_' + str(epoch))
    # assertions
    assert samples.shape[2] == settings['num_generated_features']
    # create VARIABLE Z
    Z = tf.get_variable(name='Z', shape=[num_samples, settings['seq_length'],
                        settings['latent_dim']],
                        initializer=tf.random_normal_initializer())
    if C_samples is None:
        # create outputs
        G_samples = generator(Z, settings['hidden_units_g'], settings['seq_length'],
                              num_samples, settings['num_generated_features'],
                              reuse=False, parameters=parameters)
        fd = None
    else:
        CG = tf.placeholder(tf.float32, [num_samples, settings['cond_dim']])
        assert C_samples.shape[0] == samples.shape[0]
        # CGAN
        G_samples = generator(Z, settings['hidden_units_g'], settings['seq_length'], 
                              num_samples, settings['num_generated_features'], 
                              reuse=False, parameters=parameters, cond_dim=settings['cond_dim'], c=CG)
        fd = {CG: C_samples}

    # define loss
    if heuristic_sigma is None:
        heuristic_sigma = mmd.median_pairwise_distance(samples)     # this is noisy
        print('heuristic_sigma:', heuristic_sigma)
    Kxx, Kxy, Kyy, wts = mmd._mix_rbf_kernel(G_samples, samples, sigmas=tf.constant(value=heuristic_sigma, shape=(1, 1)))
    similarity_per_sample = tf.diag_part(Kxy)
    reconstruction_error_per_sample = 1 - similarity_per_sample
    #reconstruction_error_per_sample = tf.reduce_sum((tf.nn.l2_normalize(G_samples, dim=1) - tf.nn.l2_normalize(samples, dim=1))**2, axis=[1,2])
    similarity = tf.reduce_mean(similarity_per_sample)
    reconstruction_error = 1 - similarity
    # updater
#    solver = tf.train.AdamOptimizer().minimize(reconstruction_error_per_sample, var_list=[Z])
    #solver = tf.train.RMSPropOptimizer(learning_rate=500).minimize(reconstruction_error, var_list=[Z])
    solver = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(reconstruction_error_per_sample, var_list=[Z])
    #solver = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9).minimize(reconstruction_error_per_sample, var_list=[Z])

    grad_Z = tf.gradients(reconstruction_error_per_sample, Z)[0]
    grad_per_Z = tf.norm(grad_Z, axis=(1, 2))
    grad_norm = tf.reduce_mean(grad_per_Z)
    #solver = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(reconstruction_error, var_list=[Z])
    print('Finding latent state corresponding to samples...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        error = sess.run(reconstruction_error, feed_dict=fd)
        g_n = sess.run(grad_norm, feed_dict=fd)
        print(g_n)
        i = 0
        if not n_iter is None:
            while i < n_iter:
                _ = sess.run(solver, feed_dict=fd)
                error = sess.run(reconstruction_error, feed_dict=fd)
                i += 1
        else:
            if not g_tolerance is None:
                while g_n > g_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error, g_n = sess.run([reconstruction_error, grad_norm], feed_dict=fd)
                    i += 1
                    print(error, g_n)
                    if i > max_iter:
                        break
            else:
                while np.abs(error) > e_tolerance:
                    _ = sess.run(solver, feed_dict=fd)
                    error = sess.run(reconstruction_error, feed_dict=fd)
                    i += 1
                    print(error)
                    if i > max_iter:
                        break
        Zs = sess.run(Z, feed_dict=fd)
        error_per_sample = sess.run(reconstruction_error_per_sample, feed_dict=fd)
        print('Z found in', i, 'iterations with final reconstruction error of', error)
    tf.reset_default_graph()
    return Zs, error_per_sample, heuristic_sigma

# (originally from https://github.com/tensorflow/models/tree/master/research/differential_privacy,
# possibly with some small edits by @corcra)

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Differentially private optimizers.
"""
from __future__ import division

import tensorflow as tf

from differential_privacy.dp_sgd.dp_optimizer import utils
#from differential_privacy.dp_sgd.per_example_gradients import per_example_gradients

import pdb

class DPGradientDescentOptimizer(tf.train.GradientDescentOptimizer):
  """Differentially private gradient descent optimizer.
  """

  def __init__(self, learning_rate, eps_delta, sanitizer,
               sigma=None, use_locking=False, name="DPGradientDescent",
               batches_per_lot=1):
    """Construct a differentially private gradient descent optimizer.

    The optimizer uses fixed privacy budget for each batch of training.

    Args:
      learning_rate: for GradientDescentOptimizer.
      eps_delta: EpsDelta pair for each epoch.
      sanitizer: for sanitizing the graident.
      sigma: noise sigma. If None, use eps_delta pair to compute sigma;
        otherwise use supplied sigma directly.
      use_locking: use locking.
      name: name for the object.
      batches_per_lot: Number of batches in a lot.
    """

    super(DPGradientDescentOptimizer, self).__init__(learning_rate,
                                                     use_locking, name)
    # Also, if needed, define the gradient accumulators
    self._batches_per_lot = batches_per_lot
    self._grad_accum_dict = {}
    if batches_per_lot > 1:
      self._batch_count = tf.Variable(1, dtype=tf.int32, trainable=False,
                                      name="batch_count")
      var_list = tf.trainable_variables()
      with tf.variable_scope("grad_acc_for"):
        for var in var_list:
          v_grad_accum = tf.Variable(tf.zeros_like(var),
                                     trainable=False,
                                     name=utils.GetTensorOpName(var))
          self._grad_accum_dict[var.name] = v_grad_accum

    self._eps_delta = eps_delta
    self._sanitizer = sanitizer
    self._sigma = sigma

  def compute_sanitized_gradients(self, loss, var_list=None,
                                  add_noise=True):
    """Compute the sanitized gradients.

    Args:
      loss: the loss tensor.
      var_list: the optional variables.
      add_noise: if true, then add noise. Always clip.
    Returns:
      a pair of (list of sanitized gradients) and privacy spending accumulation
      operations.
    Raises:
      TypeError: if var_list contains non-variable.
    """

    self._assert_valid_dtypes([loss])

    xs = [tf.convert_to_tensor(x) for x in var_list]
    # TODO check this change
    loss_list = tf.unstack(loss, axis=0)
    px_grads_byexample = [tf.gradients(l, xs) for l in loss_list]
    px_grads = [[x[v] for x in px_grads_byexample] for v in range(len(xs))]
    #px_grads = tf.gradients(loss, xs)
    # add a dummy 0th dimension to reflect the fact that we have a batch size of 1...
  #  px_grads = [tf.expand_dims(x, 0) for x in px_grads]
#    px_grads = per_example_gradients.PerExampleGradients(loss, xs)
    sanitized_grads = []
    for px_grad, v in zip(px_grads, var_list):
      tensor_name = utils.GetTensorOpName(v)
      sanitized_grad = self._sanitizer.sanitize(
          px_grad, self._eps_delta, sigma=self._sigma,
          tensor_name=tensor_name, add_noise=add_noise,
          num_examples=self._batches_per_lot * tf.slice(
              tf.shape(px_grad), [0], [1]))
      sanitized_grads.append(sanitized_grad)

    return sanitized_grads

  def minimize(self, loss, global_step=None, var_list=None,
               name=None):
    """Minimize using sanitized gradients.

    This gets a var_list which is the list of trainable variables.
    For each var in var_list, we defined a grad_accumulator variable
    during init. When batches_per_lot > 1, we accumulate the gradient
    update in those. At the end of each lot, we apply the update back to
    the variable. This has the effect that for each lot we compute
    gradients at the point at the beginning of the lot, and then apply one
    update at the end of the lot. In other words, semantically, we are doing
    SGD with one lot being the equivalent of one usual batch of size
    batch_size * batches_per_lot.
    This allows us to simulate larger batches than our memory size would permit.

    The lr and the num_steps are in the lot world.

    Args:
      loss: the loss tensor.
      global_step: the optional global step.
      var_list: the optional variables.
      name: the optional name.
    Returns:
      the operation that runs one step of DP gradient descent.
    """

    # First validate the var_list

    if var_list is None:
      var_list = tf.trainable_variables()
    for var in var_list:
      if not isinstance(var, tf.Variable):
        raise TypeError("Argument is not a variable.Variable: %s" % var)

    # Modification: apply gradient once every batches_per_lot many steps.
    # This may lead to smaller error

    if self._batches_per_lot == 1:
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list)

      grads_and_vars = list(zip(sanitized_grads, var_list))
      self._assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])

      apply_grads = self.apply_gradients(grads_and_vars,
                                         global_step=global_step, name=name)
      return apply_grads

    # Condition for deciding whether to accumulate the gradient
    # or actually apply it.
    # we use a private self_batch_count to keep track of number of batches.
    # global step will count number of lots processed.

    update_cond = tf.equal(tf.constant(0),
                           tf.mod(self._batch_count,
                                  tf.constant(self._batches_per_lot)))

    # Things to do for batches other than last of the lot.
    # Add non-noisy clipped grads to shadow variables.

    def non_last_in_lot_op(loss, var_list):
      """Ops to do for a typical batch.

      For a batch that is not the last one in the lot, we simply compute the
      sanitized gradients and apply them to the grad_acc variables.

      Args:
        loss: loss function tensor
        var_list: list of variables
      Returns:
        A tensorflow op to do the updates to the gradient accumulators
      """
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list, add_noise=False)

      update_ops_list = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        update_ops_list.append(grad_acc_v.assign_add(grad))
      update_ops_list.append(self._batch_count.assign_add(1))
      return tf.group(*update_ops_list)

    # Things to do for last batch of a lot.
    # Add noisy clipped grads to accumulator.
    # Apply accumulated grads to vars.

    def last_in_lot_op(loss, var_list, global_step):
      """Ops to do for last batch in a lot.

      For the last batch in the lot, we first add the sanitized gradients to
      the gradient acc variables, and then apply these
      values over to the original variables (via an apply gradient)

      Args:
        loss: loss function tensor
        var_list: list of variables
        global_step: optional global step to be passed to apply_gradients
      Returns:
        A tensorflow op to push updates from shadow vars to real vars.
      """

      # We add noise in the last lot. This is why we need this code snippet
      # that looks almost identical to the non_last_op case here.
      sanitized_grads = self.compute_sanitized_gradients(
          loss, var_list=var_list, add_noise=True)

      normalized_grads = []
      for var, grad in zip(var_list, sanitized_grads):
        grad_acc_v = self._grad_accum_dict[var.name]
        # To handle the lr difference per lot vs per batch, we divide the
        # update by number of batches per lot.
        normalized_grad = tf.div(grad_acc_v.assign_add(grad),
                                 tf.to_float(self._batches_per_lot))

        normalized_grads.append(normalized_grad)

      with tf.control_dependencies(normalized_grads):
        grads_and_vars = list(zip(normalized_grads, var_list))
        self._assert_valid_dtypes(
            [v for g, v in grads_and_vars if g is not None])
        apply_san_grads = self.apply_gradients(grads_and_vars,
                                               global_step=global_step,
                                               name="apply_grads")

      # Now reset the accumulators to zero
      resets_list = []
      with tf.control_dependencies([apply_san_grads]):
        for _, acc in self._grad_accum_dict.items():
          reset = tf.assign(acc, tf.zeros_like(acc))
          resets_list.append(reset)
      resets_list.append(self._batch_count.assign_add(1))

      last_step_update = tf.group(*([apply_san_grads] + resets_list))
      return last_step_update
    # pylint: disable=g-long-lambda
    update_op = tf.cond(update_cond,
                        lambda: last_in_lot_op(
                            loss, var_list,
                            global_step),
                        lambda: non_last_in_lot_op(
                            loss, var_list))
    return tf.group(update_op)
  
  from __future__ import division

import math

import numpy
import tensorflow as tf


class LayerParameters(object):
  """class that defines a non-conv layer."""
  def __init__(self):
    self.name = ""
    self.num_units = 0
    self._with_bias = False
    self.relu = False
    self.gradient_l2norm_bound = 0.0
    self.bias_gradient_l2norm_bound = 0.0
    self.trainable = True
    self.weight_decay = 0.0


class ConvParameters(object):
  """class that defines a conv layer."""
  def __init__(self):
    self.patch_size = 5
    self.stride = 1
    self.in_channels = 1
    self.out_channels = 0
    self.with_bias = True
    self.relu = True
    self.max_pool = True
    self.max_pool_size = 2
    self.max_pool_stride = 2
    self.trainable = False
    self.in_size = 28
    self.name = ""
    self.num_outputs = 0
    self.bias_stddev = 0.1


# Parameters for a layered neural network.
class NetworkParameters(object):
  """class that define the overall model structure."""
  def __init__(self):
    self.input_size = 0
    self.projection_type = 'NONE'  # NONE, RANDOM, PCA
    self.projection_dimensions = 0
    self.default_gradient_l2norm_bound = 0.0
    self.layer_parameters = []  # List of LayerParameters
    self.conv_parameters = []  # List of ConvParameters


def GetTensorOpName(x):
  """Get the name of the op that created a tensor.

  Useful for naming related tensors, as ':' in name field of op is not permitted

  Args:
    x: the input tensor.
  Returns:
    the name of the op.
  """

  t = x.name.rsplit(":", 1)
  if len(t) == 1:
    return x.name
  else:
    return t[0]


def BuildNetwork(inputs, network_parameters):
  """Build a network using the given parameters.

  Args:
    inputs: a Tensor of floats containing the input data.
    network_parameters: NetworkParameters object
      that describes the parameters for the network.
  Returns:
    output, training_parameters: where the outputs (a tensor) is the output
      of the network, and training_parameters (a dictionary that maps the
      name of each variable to a dictionary of parameters) is the parameters
      used during training.
  """

  training_parameters = {}
  num_inputs = network_parameters.input_size
  outputs = inputs
  projection = None

  # First apply convolutions, if needed
  for conv_param in network_parameters.conv_parameters:
    outputs = tf.reshape(
        outputs,
        [-1, conv_param.in_size, conv_param.in_size,
         conv_param.in_channels])
    conv_weights_name = "%s_conv_weight" % (conv_param.name)
    conv_bias_name = "%s_conv_bias" % (conv_param.name)
    conv_std_dev = 1.0 / (conv_param.patch_size
                          * math.sqrt(conv_param.in_channels))
    conv_weights = tf.Variable(
        tf.truncated_normal([conv_param.patch_size,
                             conv_param.patch_size,
                             conv_param.in_channels,
                             conv_param.out_channels],
                            stddev=conv_std_dev),
        trainable=conv_param.trainable,
        name=conv_weights_name)
    conv_bias = tf.Variable(
        tf.truncated_normal([conv_param.out_channels],
                            stddev=conv_param.bias_stddev),
        trainable=conv_param.trainable,
        name=conv_bias_name)
    training_parameters[conv_weights_name] = {}
    training_parameters[conv_bias_name] = {}
    conv = tf.nn.conv2d(outputs, conv_weights,
                        strides=[1, conv_param.stride,
                                 conv_param.stride, 1],
                        padding="SAME")
    relud = tf.nn.relu(conv + conv_bias)
    mpd = tf.nn.max_pool(relud, ksize=[1,
                                       conv_param.max_pool_size,
                                       conv_param.max_pool_size, 1],
                         strides=[1, conv_param.max_pool_stride,
                                  conv_param.max_pool_stride, 1],
                         padding="SAME")
    outputs = mpd
    num_inputs = conv_param.num_outputs
    # this should equal
    # in_size * in_size * out_channels / (stride * max_pool_stride)

  # once all the convs are done, reshape to make it flat
  outputs = tf.reshape(outputs, [-1, num_inputs])

  # Now project, if needed
  if network_parameters.projection_type is not "NONE":
    projection = tf.Variable(tf.truncated_normal(
        [num_inputs, network_parameters.projection_dimensions],
        stddev=1.0 / math.sqrt(num_inputs)), trainable=False, name="projection")
    num_inputs = network_parameters.projection_dimensions
    outputs = tf.matmul(outputs, projection)

  # Now apply any other layers

  for layer_parameters in network_parameters.layer_parameters:
    num_units = layer_parameters.num_units
    hidden_weights_name = "%s_weight" % (layer_parameters.name)
    hidden_weights = tf.Variable(
        tf.truncated_normal([num_inputs, num_units],
                            stddev=1.0 / math.sqrt(num_inputs)),
        name=hidden_weights_name, trainable=layer_parameters.trainable)
    training_parameters[hidden_weights_name] = {}
    if layer_parameters.gradient_l2norm_bound:
      training_parameters[hidden_weights_name]["gradient_l2norm_bound"] = (
          layer_parameters.gradient_l2norm_bound)
    if layer_parameters.weight_decay:
      training_parameters[hidden_weights_name]["weight_decay"] = (
          layer_parameters.weight_decay)

    outputs = tf.matmul(outputs, hidden_weights)
    if layer_parameters.with_bias:
      hidden_biases_name = "%s_bias" % (layer_parameters.name)
      hidden_biases = tf.Variable(tf.zeros([num_units]),
                                  name=hidden_biases_name)
      training_parameters[hidden_biases_name] = {}
      if layer_parameters.bias_gradient_l2norm_bound:
        training_parameters[hidden_biases_name][
            "bias_gradient_l2norm_bound"] = (
                layer_parameters.bias_gradient_l2norm_bound)

      outputs += hidden_biases
    if layer_parameters.relu:
      outputs = tf.nn.relu(outputs)
    # num_inputs for the next layer is num_units in the current layer.
    num_inputs = num_units

  return outputs, projection, training_parameters


def VaryRate(start, end, saturate_epochs, epoch):
  """Compute a linearly varying number.

  Decrease linearly from start to end until epoch saturate_epochs.

  Args:
    start: the initial number.
    end: the end number.
    saturate_epochs: after this we do not reduce the number; if less than
      or equal to zero, just return start.
    epoch: the current learning epoch.
  Returns:
    the caculated number.
  """
  if saturate_epochs <= 0:
    return start

  step = (start - end) / (saturate_epochs - 1)
  if epoch < saturate_epochs:
    return start - step * epoch
  else:
    return end


def BatchClipByL2norm(t, upper_bound, name=None):
  """Clip an array of tensors by L2 norm.

  Shrink each dimension-0 slice of tensor (for matrix it is each row) such
  that the l2 norm is at most upper_bound. Here we clip each row as it
  corresponds to each example in the batch.

  Args:
    t: the input tensor.
    upper_bound: the upperbound of the L2 norm.
    name: optional name.
  Returns:
    the clipped tensor.
  """

  assert upper_bound > 0
  with tf.name_scope(values=[t, upper_bound], name=name,
                     default_name="batch_clip_by_l2norm") as name:
    saved_shape = tf.shape(t)
    batch_size = tf.slice(saved_shape, [0], [1])
    t2 = tf.reshape(t, tf.concat(axis=0, values=[batch_size, [-1]]))
    upper_bound_inv = tf.fill(tf.slice(saved_shape, [0], [1]),
                              tf.constant(1.0/upper_bound))
    # Add a small number to avoid divide by 0
    l2norm_inv = tf.rsqrt(tf.reduce_sum(t2 * t2, [1]) + 0.000001)
    scale = tf.minimum(l2norm_inv, upper_bound_inv) * upper_bound
    clipped_t = tf.matmul(tf.diag(scale), t2)
    clipped_t = tf.reshape(clipped_t, saved_shape, name=name)
  return clipped_t


def SoftThreshold(t, threshold_ratio, name=None):
  """Soft-threshold a tensor by the mean value.

  Softthreshold each dimension-0 vector (for matrix it is each column) by
  the mean of absolute value multiplied by the threshold_ratio factor. Here
  we soft threshold each column as it corresponds to each unit in a layer.

  Args:
    t: the input tensor.
    threshold_ratio: the threshold ratio.
    name: the optional name for the returned tensor.
  Returns:
    the thresholded tensor, where each entry is soft-thresholded by
    threshold_ratio times the mean of the aboslute value of each column.
  """

  assert threshold_ratio >= 0
  with tf.name_scope(values=[t, threshold_ratio], name=name,
                     default_name="soft_thresholding") as name:
    saved_shape = tf.shape(t)
    t2 = tf.reshape(t, tf.concat(axis=0, values=[tf.slice(saved_shape, [0], [1]), -1]))
    t_abs = tf.abs(t2)
    t_x = tf.sign(t2) * tf.nn.relu(t_abs -
                                   (tf.reduce_mean(t_abs, [0],
                                                   keep_dims=True) *
                                    threshold_ratio))
    return tf.reshape(t_x, saved_shape, name=name)


def AddGaussianNoise(t, sigma, name=None):
  """Add i.i.d. Gaussian noise (0, sigma^2) to every entry of t.

  Args:
    t: the input tensor.
    sigma: the stddev of the Gaussian noise.
    name: optional name.
  Returns:
    the noisy tensor.
  """

  with tf.name_scope(values=[t, sigma], name=name,
                     default_name="add_gaussian_noise") as name:
    noisy_t = t + tf.random_normal(tf.shape(t), stddev=sigma)
  return noisy_t


def GenerateBinomialTable(m):
  """Generate binomial table.

  Args:
    m: the size of the table.
  Returns:
    A two dimensional array T where T[i][j] = (i choose j),
    for 0<= i, j <=m.
  """

  table = numpy.zeros((m + 1, m + 1), dtype=numpy.float64)
  for i in range(m + 1):
    table[i, 0] = 1
  for i in range(1, m + 1):
    for j in range(1, m + 1):
      v = table[i - 1, j] + table[i - 1, j -1]
      assert not math.isnan(v) and not math.isinf(v)
      table[i, j] = v
  return tf.convert_to_tensor(table)