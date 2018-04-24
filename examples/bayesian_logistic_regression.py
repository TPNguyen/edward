"""Bayesian logistic regression using Hamiltonian Monte Carlo.

We visualize the fit.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

from matplotlib import pyplot as plt
from edward.models import Bernoulli, Normal, Empirical
#from edward.criticisms import ess

flags = tf.app.flags

flags.DEFINE_integer("N", default=40, help="Number of data points.")
flags.DEFINE_integer("D", default=1, help="Number of features.")
flags.DEFINE_integer("T", default=5000, help="Number of posterior samples.")
flags.DEFINE_string("logdir", default="log", help="The directory to save log")

# add a new flag to pass in sampler
flags.DEFINE_string("sampler", default="HMC", help="The sampler to use, HMC(Hamiltonian Monte Carlo), SGHMC(Stochastic Hamiltonian Monte Carlo), SGLD(Stochastic gradient Langevin dynamics)")

FLAGS = flags.FLAGS

def mcse(traces, g = None):
    if (not g):
        g = lambda x: x
    n = traces.shape[0]
    b = int(math.sqrt(n)) # batch size
    a = int(n/b)
    
    # apply batch means
    mu_hat = np.mean(g(traces))
    batch_means = [np.mean(g(traces[i*b:i*b + a, ])) for i in range(a)]
    batch_means = np.array(batch_means, dtype = np.float128)
    var_hat = sum((batch_means - mu_hat)**2)*b/(a-1)
    se = math.sqrt(var_hat/n) 
    return var_hat

def ess(traces, g = None):
    iid_var = np.var(traces, axis = 0, dtype = np.float64)
    print(iid_var)
    sigma = mcse(traces, g)
    print(sigma)
    return (iid_var/sigma)*(traces.shape[0])





def build_toy_dataset(N, noise_std=0.1):
  D = 1
  X = np.linspace(-6, 6, num=N)
  y = np.tanh(X) + np.random.normal(0, noise_std, size=N)
  y[y < 0.5] = 0
  y[y >= 0.5] = 1
  X = (X - 4.0) / 4.0
  X = X.reshape((N, D))
  return X, y

    # test_val = tf.placeholder(tf.float32, name='tmp1')
    # summary_1 = tf.summary.scalar('tmp1', test_val)

    # test_val2 = tf.placeholder(tf.float32, name='tmp2')
    # summary_2 = tf.summary.scalar('tmp2', test_val2)

    # sess = tf.InteractiveSession()

    # train_writer = tf.summary.FileWriter('be_polite', sess.graph)
    # tf.global_variables_initializer().run()

    # summary, val1 = sess.run([summary_1, test_val], feed_dict={'tmp1:0': 1.0})
    # train_writer.add_summary(summary)

def main(_):
  ed.set_seed(42)

  # DATA
  X_train, y_train = build_toy_dataset(FLAGS.N)

  # MODEL
  X = tf.placeholder(tf.float32, [FLAGS.N, FLAGS.D])
  w = Normal(loc=tf.zeros(FLAGS.D), scale=3.0 * tf.ones(FLAGS.D))
  b = Normal(loc=tf.zeros([]), scale=3.0 * tf.ones([]))
  y = Bernoulli(logits=ed.dot(X, w) + b)

  # INFERENCE
  qw = Empirical(params=tf.get_variable("qw/params", [FLAGS.T, FLAGS.D]))
  qb = Empirical(params=tf.get_variable("qb/params", [FLAGS.T]))

  if (FLAGS.sampler == "SGHMC"):
    inference = ed.SGHMC({w: qw, b: qb}, data={X: X_train, y: y_train})
  elif (FLAGS.sampler == "SGLD"):
    inference = ed.SGLD({w: qw, b: qb}, data={X: X_train, y: y_train})
  elif (FLAGS.sampler == "HMC"):
    inference = ed.HMC({w: qw, b: qb}, data={X: X_train, y: y_train})

  # qb_ess = tf.placeholder(tf.float64, name='qb_ess')
  # qb_ess_summary_1 = tf.summary.scalar('qb_ess', qb_ess)

  qb_ess = tf.Variable(3, trainable=False, name="qb_ess", dtype = tf.float64)

  summary_writer = tf.summary.FileWriter(FLAGS.logdir)
  # tf.get_variable('qb_ess', shape = [1], dtype = tf.float64)
  tf.summary.scalar('qb_ess', qb_ess)
  inference.summarize = tf.summary.merge_all()

  inference.initialize(n_print=10, step_size=0.6, logdir=FLAGS.logdir)





  # Alternatively, use variational inference.
  # qw_loc = tf.get_variable("qw_loc", [FLAGS.D])
  # qw_scale = tf.nn.softplus(tf.get_variable("qw_scale", [FLAGS.D]))
  # qb_loc = tf.get_variable("qb_loc", []) + 10.0
  # qb_scale = tf.nn.softplus(tf.get_variable("qb_scale", []))

  # qw = Normal(loc=qw_loc, scale=qw_scale)
  # qb = Normal(loc=qb_loc, scale=qb_scale)

  # inference = ed.KLqp({w: qw, b: qb}, data={X: X_train, y: y_train})
  # inference.initialize(n_print=10, n_iter=600)



  tf.global_variables_initializer().run()


  # CRITICISM
  sess = ed.get_session()


  # Set up figure.
  fig = plt.figure(figsize=(8, 8), facecolor='white')
  ax = fig.add_subplot(111, frameon=False)
  plt.ion()
  plt.show(block=False)

  # Build samples from inferred posterior.
  n_samples = 50
  inputs = np.linspace(-5, 3, num=400, dtype=np.float32).reshape((400, 1))
  probs = tf.stack([tf.sigmoid(ed.dot(inputs, qw.sample()) + qb.sample())
                    for _ in range(n_samples)])





  for t in range(5000):
    info_dict = inference.update()
    inference.print_progress(info_dict)

    trace2 = sess.run(qb.params)

    qb_ess = ess(trace2)
    print("effective sample size")
    print(qb_ess)
    #qb_ess = 10.5
    #summary_writer.add_summary(qb_ess, t)
    # summary, val1 = sess.run([qb_ess], feed_dict={'qb_ess': 1.0})
    # summary_writer.add_summary(summary)

    


    if t % inference.n_print == 0:
      outputs = probs.eval()
      # print("start")
      # print(outputs.shape)
      # print(type(outputs))
      # print(outputs[0])
      # print(outputs[0][0])
      # print("end")

      # Plot data and functions
      plt.cla()
      ax.plot(X_train[:], y_train, 'bx')
      for s in range(n_samples):
        ax.plot(inputs[:], outputs[s], alpha=0.2)

      ax.set_xlim([-5, 3])
      ax.set_ylim([-0.5, 1.5])
      plt.draw()
      plt.pause(1.0 / 60.0)

if __name__ == "__main__":
  plt.style.use("ggplot")
  tf.app.run()
