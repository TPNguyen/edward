import edward as ed
import numpy as np
import tensorflow as tf
from edward.models import Bernoulli, Empirical, Normal
import matplotlib.pyplot as plt
import math


N = 50000  # number of data points
D = 1  # number of features
T = 1000  # number of empirical samples 

def build_toy_dataset(N, noise_std=0.1):
  x = np.random.normal(loc=0.0, scale = 1.0, size=N) 

  # b = 1.0  
  # b = np.array((np.float64)(b))
  # print(x.shape)
  # print(b.shape)
  y = x + np.random.normal(0, noise_std, size=N) # b = 1 
  y = Bernoulli(logits=y) 
  x = x.reshape((N, D))
  return x, y




# DATA
# x_data = np.zeros([N, D])
# y_data = np.zeros([N])

x_data, y_data = build_toy_dataset(N, 0.1)

# MODEL
x = tf.Variable(x_data, trainable=False, dtype = tf.float32)
# print(type(x))
beta = Normal(loc=tf.zeros(D), scale=tf.ones(D))
# print(type(beta))
y = Bernoulli(logits=ed.dot(x, beta))

# INFERENCE
qbeta = Empirical(params=tf.Variable(tf.zeros([T, D])))
inference = ed.HMC({beta: qbeta}, data={y: y_data})
inference.run(step_size=10/N, n_steps=10)

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


qw_ess = ess(trace_beta)  
print(qw_ess)

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
  
  
