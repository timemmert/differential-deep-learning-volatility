import jax.numpy as jnp


def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x ** a


def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    sides = 1. / ((1. * a + 1) * n ** (1. * a + 1))
    return jnp.array([
        [1. / n, sides],
        [sides, 1. / ((2. * a + 1) * n ** (2. * a + 1))]
    ])
