# WOndering what is happening here? https://github.com/ryanmccrickerd/rough_bergomi/blob/master/notebooks/rbergomi.ipynb
from functools import partial

import jax
import numpy as np
from jax import jit

from util import b, cov, g
import jax.numpy as jnp


def compute_dW1(e, c, m, s, rng_key):
    """
    Produces random numbers for variance process with required
    covariance structure.
    """
    return jax.random.multivariate_normal(key=rng_key, mean=e, cov=c, shape=(m, s))


def compute_dW2(m, s, dt, rng_key):
    """
    Obtain orthogonal increments.
    """
    return jax.random.normal(key=rng_key, shape=(m, s)) * np.sqrt(dt)


def compute_Y(m, s, a, n, dW):
    """
    Constructs Volterra process from appropriately
    correlated 2d Brownian increments.
    """

    # Construct Y1 through exact integral
    Y1 = jnp.concatenate((
        jnp.zeros((m, 1)),
        dW[:, :, 1]
    ), axis=1)  # Exact integrals

    ks = jnp.arange(2, 1 + s, 1)
    G = jnp.concatenate((np.array([0, 0]), g(b(ks, a) / n, a)))

    X = dW[:, :, 0]  # Xi

    # Compute convolution, FFT not used for small n
    # Possible to compute for all paths in C-layer?
    GX = jnp.apply_along_axis(func1d=lambda x: jnp.convolve(G, x), arr=X, axis=1)

    # Extract appropriate part of convolution
    Y2 = GX[:, :1 + s]  # Riemann sums

    # Finally contruct and return full process
    Y = jnp.sqrt(2 * a + 1) * (Y1 + Y2)
    return Y


def compute_dB(dW1, dW2, rho=0.0):
    """
    Constructs correlated price Brownian increments, dB.
    """
    dB = rho * dW1[:, :, 0] + jnp.sqrt(1 - rho ** 2) * dW2
    return dB


def compute_V(a, t, Y, xi=1.0, eta=1.0):
    """
    rBergomi variance process.
    """
    V = xi * jnp.exp(eta * Y - 0.5 * eta ** 2 * t ** (2 * a + 1))
    return V


def compute_S(m, dt, V, dB, S0=1):
    """
    rBergomi price process.
    """

    # Construct non-anticipative Riemann increments
    increments = jnp.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

    # Cumsum is a little slower than Python loop.
    integral = jnp.cumsum(increments, axis=1)

    return S0 * jnp.concatenate(
        (
            jnp.ones((m, 1,)),
            jnp.exp(integral)
        ),
        axis=1
    )


def S1(dt, V, dW1, rho, S0=1):
    """
    rBergomi parallel price process.
    """
    # Construct non-anticipative Riemann increments
    increments = rho * np.sqrt(V[:, :-1]) * dW1[:, :, 0] - 0.5 * rho ** 2 * V[:, :-1] * dt

    # Cumsum is a little slower than Python loop.
    integral = np.cumsum(increments, axis=1)

    S = np.zeros_like(V)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(integral)
    return S


def price_single_call_option_given_paths(dt, paths, strike_price, maturity):
    maturity_index = (maturity / dt).astype(int)
    res = jnp.greater(paths[:, maturity_index], strike_price) * paths[:, maturity_index]
    return 1 / paths.shape[0] * res.sum(axis=0)


@partial(jit, static_argnums=(0, 1, 2, 3))
def price(m, s, n, dt, t, S0, a, rho, eta, xi, strike_price, maturity, multipliers):
    paths = simulate_paths(m, s, n, dt, t, S0, a, rho, eta, xi)
    return price_single_call_option_given_paths(dt, paths, strike_price, maturity) * multipliers


def price_and_grad_batch(m, s, n, dt, t, S0, a, rho, eta, xi, strike_prices, maturities):
    # cross strike prices and maturities
    strikes_new = jnp.repeat(strike_prices, maturities.shape[0], axis=None)
    maturities_new = jnp.tile(maturities, strike_prices.shape[0])

    paths = simulate_paths(m, s, n, dt, t, S0, a, rho, eta, xi)
    #jac_paths = jax.jacfwd(simulate_paths, argnums=(6, 7, 8, 9))(m, s, n, dt, t, S0, a, rho, eta, xi)
    #res = jax.vmap(price_single_call_option_given_paths, in_axes=(None, None, 0, 0))(dt, paths, strikes_new, maturities_new)

    jacobian = jax.jacfwd(
        price,
        argnums=(6, 7, 8, 9, 10, 11, 12)
    )(m, s, n, dt, t, S0, a, rho, eta, xi, 1., 1., jnp.array([1., 2.]))

    y, dy_dx = jax.vmap(
        jax.value_and_grad(
            price,
            argnums=(6, 7, 8, 9, 10, 11)
        ),
        in_axes=(None, None, None, None, None, None, None, None, None, None, 0, 0),
        out_axes=(0, 0)
    )(m, s, n, dt, t, S0, a, rho, eta, xi, strikes_new, maturities_new)
    # x is an array of all inputs that vmap took
    x_constants = jnp.array([[a, rho, eta, xi]] * strikes_new.shape[0])
    x = jnp.concatenate((x_constants, jnp.expand_dims(strikes_new, 1), jnp.expand_dims(maturities_new, 1)), axis=1)
    return x, y, dy_dx


@partial(jit, static_argnums=(0, 1, 2, 3))
def simulate_paths(m, s, n, dt, t, S0, a, rho, eta, xi):
    e = jnp.array([0, 0])
    c = cov(a, n)

    dW1 = compute_dW1(e=e, c=c, m=m, s=s, rng_key=jax.random.PRNGKey(123))
    dW2 = compute_dW2(m=m, s=s, dt=dt, rng_key=jax.random.PRNGKey(123123))

    Y = compute_Y(m=m, s=s, a=a, n=n, dW=dW1)
    dB = compute_dB(dW1=dW1, dW2=dW2, rho=rho)
    V = compute_V(a=a, t=t, Y=Y, xi=xi, eta=eta)
    S = compute_S(m=m, dt=dt, V=V, dB=dB, S0=S0)
    return S
