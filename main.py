import jax.random

import jax.numpy as jnp
from rbergomi import price_and_grad_batch

T = 1.0
n = 100
m = 30000
S0 = 1.
initial_key = jax.random.PRNGKey(0)
key_a, key_xi, key_rho, key_eta = jax.random.split(initial_key, 4)

n_samples = 10

minval_a = -0.5
maxval_a = 0.5
a = jax.random.uniform(key_a, (n_samples,), minval=minval_a, maxval=maxval_a)

minval_xi = 0.2 ** 2
maxval_xi = 0.5 ** 2
xi = jax.random.uniform(key_xi, (n_samples,), minval=minval_xi, maxval=maxval_xi)

minval_rho = -0.9
maxval_rho = -0.1
rho = jax.random.uniform(key_rho, (n_samples,), minval=minval_rho, maxval=maxval_rho)

minval_eta = 0.5
maxval_eta = 4
eta = jax.random.uniform(key_rho, (n_samples,), minval=minval_eta, maxval=maxval_eta)

dt = 1.0 / n
s = int(n * T)  # Steps
t = jnp.linspace(0, T, 1 + s)[jnp.newaxis, :]  # Time grid

strikes = jnp.array([1.])
maturities = jnp.array([0.5, 1])

price_and_grad_batch(m, s, n, dt, t, S0, a[0], rho[0], eta[0], xi[0], strikes, maturities)
x, y, dy_dx = jax.vmap(
    price_and_grad_batch,
    in_axes=(None, None, None, None, None, None, 0, 0, 0, 0, None, None)
)(m, s, n, dt, t, S0, a, rho, eta, xi, strikes, maturities)
print("Done")