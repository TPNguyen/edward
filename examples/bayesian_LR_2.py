import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Bernoulli, Empirical, Normal
import matplotlib.pyplot as plt
import math

flags = tf.app.flags

flags.DEFINE_integer("N", default=50000, help="Number of data points.")
flags.DEFINE_integer("D", default=1, help="Number of features.")
flags.DEFINE_integer("T", default=2000, help="Number of posterior samples.")
flags.DEFINE_string("logdir", default="log", help="The directory to save log")

# add a new flag to pass in sampler
flags.DEFINE_string("sampler", default="HMC", help="The sampler to use, HMC(Hamiltonian Monte Carlo), SGHMC(Stochastic Hamiltonian Monte Carlo), SGLD(Stochastic gradient Langevin dynamics)")


FLAGS = flags.FLAGS



def build_toy_dataset(noise_std=0.1):
  x = np.random.normal(loc=0.0, scale = 1.0, size=FLAGS.N*FLAGS.D) 
  x = x.reshape((FLAGS.N, FLAGS.D)) 
  b = np.linspace(1.0, 10.0, num=FLAGS.D)

  # b = 1.0  
  # b = np.array((np.float64)(b))
  # print(x.shape)
  # print(b.shape)
  y = ed.dot(x, b) + np.random.normal(0, noise_std, size=FLAGS.N) # b = 1 
  y = Bernoulli(logits=y) 
  return x, y




# DATA
# x_data = np.zeros([N, D])
# y_data = np.zeros([N])

x_data, y_data = build_toy_dataset(0.1)

# MODEL
x = tf.Variable(x_data, trainable=False, dtype = tf.float32)
# print(type(x))
beta = Normal(loc=tf.zeros(FLAGS.D), scale=tf.ones(FLAGS.D))
# print(type(beta))
y = Bernoulli(logits=ed.dot(x, beta))

# INFERENCE
qbeta = Empirical(params=tf.Variable(tf.zeros([FLAGS.T, FLAGS.D])))

if (FLAGS.sampler == "SGHMC"):
  inference = ed.SGHMC({beta: qbeta}, data={y: y_data})
elif (FLAGS.sampler == "SGLD"):
  inference = ed.SGLD({beta: qbeta}, data={y: y_data})
elif (FLAGS.sampler == "HMC"):
  inference = ed.HMC({beta: qbeta}, data={y: y_data})

# inference.run(step_size=10/N, n_steps=10)
inference.run(step_size=10/FLAGS.N, n_steps=10) 


sess = ed.get_session() 
trace_beta = sess.run(qbeta.params)

# ESS 
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
  # print(iid_var)
  sigma = mcse(traces, g)
  # print(sigma)
  return (iid_var/sigma)*(traces.shape[0])


beta_ess = ess(trace_beta)  
print("Effect Sampling Size is: ")
print(beta_ess)

# traceb = sess.run(qb.params)
# print(type(trace_w))
# print(traceb.shape)
# qb_ess = ess(traceb)
# print(type(qb_ess))
# print(qb_ess.shape)
# print("effective sample size")
# print(qb_ess) 


plt.figure(1)
plt.subplot(211)
plt.plot(trace_beta)
plt.show() 
  
  
