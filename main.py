# WOndering what is happening here? https://github.com/ryanmccrickerd/rough_bergomi/blob/master/notebooks/rbergomi.ipynb
import jax
import numpy as np
from matplotlib import pyplot as plt

from util import cov
from rbergomi import dW1, dW2, Y, dB, V, S

T = 1.0
n = 100
m = 30000
S0=1

a = -0.43
xi = 0.235 ** 2
rho = -0.9
eta = 1.9

dt = 1.0 / n
s = int(n * T)  # Steps
t = np.linspace(0, T, 1 + s)[np.newaxis, :]  # Time grid

# Construct hybrid scheme correlation structure for kappa = 1
e = np.array([0, 0])
c = cov(a, n)


# np.random.seed(0)

# TODO: Remove asarray
dW1 = np.asarray(dW1(e=e, c=c, m=m, s=s, rng_key=jax.random.PRNGKey(123)))
dW2 = np.asarray(dW2(m=m, s=s, dt=dt, rng_key=jax.random.PRNGKey(123123)))

Y = Y(m=m, s=s, a=a, n=n, dW=dW1)
dB = dB(dW1=dW1, dW2=dW2, rho=rho)
V = V(a=a, t=t, Y=Y, xi=xi, eta=eta)
S = S(dt=dt, V=V, dB=dB, S0=S0)

paths_plot = 10
plot, axes = plt.subplots()
axes.plot(t[0, :], np.mean(S, axis=0), 'r')
axes.plot(t[0, :], np.ones_like(t[0, :]), 'g')

if paths_plot > 0:
    axes.plot(t[0, :], np.transpose(S[:paths_plot, :]), lw=0.5)
axes.set_xlabel(r'$t$', fontsize=16)
axes.set_ylabel(r'$S_t$', fontsize=16)
plt.grid(True)
plt.show()
