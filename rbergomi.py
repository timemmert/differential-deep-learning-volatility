import jax
import numpy as np

from util import b, g


def dW1(e, c, m, s, rng_key):
    """
    Produces random numbers for variance process with required
    covariance structure.
    """
    # np.random.multivariate_normal(e, c, (m, s))
    return jax.random.multivariate_normal(key=rng_key, mean=e, cov=c, shape=(m, s))


def dW2(m, s, dt, rng_key):
    """
    Obtain orthogonal increments.
    """
    # return np.random.randn(m, s) * np.sqrt(dt)
    return jax.random.normal(key=rng_key, shape=(m, s)) * np.sqrt(dt)


def Y(m, s, a, n, dW):
    """
    Constructs Volterra process from appropriately
    correlated 2d Brownian increments.
    """
    Y1 = np.zeros((m, 1 + s))  # Exact integrals
    Y2 = np.zeros((m, 1 + s))  # Riemann sums

    """
    for i in np.arange(1, 1 + s, 1):
        Y1[:, i] = dW[:, i - 1, 1]  # Assumes kappa = 1
    """


    # Construct Y1 through exact integral
    Y1 = np.concatenate((
        np.zeros((m, 1)),
        dW[:, :, 1]
    ), axis=1)

    # Construct arrays for convolution
    G = np.zeros(1 + s)  # Gamma
    for k in np.arange(2, 1 + s, 1):
        G[k] = g(b(k, a) / n, a)

    X = dW[:, :, 0]  # Xi

    # Initialise convolution result, GX
    GX = np.zeros((m, len(X[0, :]) + len(G) - 1))

    # Compute convolution, FFT not used for small n
    # Possible to compute for all paths in C-layer?
    for i in range(m):
        GX[i, :] = np.convolve(G, X[i, :])

    # Extract appropriate part of convolution
    Y2 = GX[:, :1 + s]

    # Finally contruct and return full process
    Y = np.sqrt(2 * a + 1) * (Y1 + Y2)
    return Y




def dB(dW1, dW2, rho=0.0):
    """
    Constructs correlated price Brownian increments, dB.
    """
    dB = rho * dW1[:, :, 0] + np.sqrt(1 - rho ** 2) * dW2
    return dB


def V(a, t, Y, xi=1.0, eta=1.0):
    """
    rBergomi variance process.
    """
    V = xi * np.exp(eta * Y - 0.5 * eta ** 2 * t ** (2 * a + 1))
    return V


def S(dt, V, dB, S0=1):
    """
    rBergomi price process.
    """

    # Construct non-anticipative Riemann increments
    increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt

    # Cumsum is a little slower than Python loop.
    integral = np.cumsum(increments, axis=1)

    S = np.zeros_like(V)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(integral)
    return S


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
