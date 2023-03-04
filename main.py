# WOndering what is happening here? https://github.com/ryanmccrickerd/rough_bergomi/blob/master/notebooks/rbergomi.ipynb
import jax
import numpy as np
from jax import grad, value_and_grad
from matplotlib import pyplot as plt

import jax.numpy as jnp
from rbergomi import simulate

T = 1.0
n = 100
m = 30000
S0 = 1.

a = -0.43
xi = 0.235 ** 2
rho = -0.9
eta = 1.9

dt = 1.0 / n
s = int(n * T)  # Steps
t = jnp.linspace(0, T, 1 + s)[np.newaxis, :]  # Time grid

# Construct hybrid scheme correlation structure for kappa = 1

K = 1.3

val_grad_sim = value_and_grad(simulate, argnums=(5, 6, 7, 8, 9, 10))

for i in range(10000):
    S, gradient = val_grad_sim(m, s, n, dt, t, K, S0, a, rho, eta, xi)
    print(i)
"""paths_plot = 10
plot, axes = plt.subplots()
axes.plot(t[0, :], np.mean(S, axis=0), 'r')
axes.plot(t[0, :], np.ones_like(t[0, :]), 'g')

if paths_plot > 0:
    axes.plot(t[0, :], np.transpose(S[:paths_plot, :]), lw=0.5)
axes.set_xlabel(r'$t$', fontsize=16)
axes.set_ylabel(r'$S_t$', fontsize=16)
plt.grid(True)
plt.show()"""
