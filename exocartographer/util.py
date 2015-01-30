import numpy as np

def logit(x, low=0, high=1):
    r"""Returns the logit function at ``x``.  Logit is defined as

    .. math::

      \mathrm{logit}\, x \equiv \log(x - \mathrm{low}) - \log(\mathrm{high} - x)

    """
    return np.log(x-low) - np.log(high-x)

def inv_logit(y, low=0, high=1):
    """Returns the ``x`` such that ``y == logit(x, low, high)``.

    """
    ey = np.exp(y)
    return low/(1.0 + ey) + high/(1.0 + 1.0/ey)

def flat_logit_log_prior(y, low=0, high=1):
    """Returns the log probability of a density that is flat in ``x`` when
    ``y = logit(x, low, high)``.

    """
    
    return np.log(high-low) + y - 2.0*np.log1p(np.exp(y))

def find_one_sigma_equivalent(logpost, x0, l0=None, dx_max=None):
    """Returns an ``(ndim,)`` shaped array giving the distance in each
    coordinate over which the log-posterior drops by 0.5 (for a
    Gaussian, this would be the one-sigma range in each coordinate).

    """
    x0 = np.atleast_1d(x0)
    sigmas = np.zeros(x0.shape[0])

    if dx_max is None:
        dx_max = 0*x0 + 1.0

    if l0 is None:
        l0 = logpost(x0)
        
    for i in range(x0.shape[0]):
        xlow = x0.copy()
        xlow[i] -= dx_max[i]
        xhigh = x0.copy()
        xhigh[i] += dx_max[i]
        llow = logpost(xlow)
        lhigh = logpost(xhigh)

        while llow < l0 - 0.5 and not xlow[i] == x0[i]:
            dx = x0[i] - xlow[i]
            xlow = x0.copy()
            xlow[i] = x0[i] - dx/2
            llow = logpost(xlow)

        while lhigh < l0 - 0.5 and not xhigh[i] == x0[i]:
            dx = xhigh[i] - x0[i]
            xhigh = x0.copy()
            xhigh[i] = x0[i] + dx/2
            lhigh = logpost(xhigh)

        sigmas[i] = 0.5*(xhigh[i] - xlow[i])

    return sigmas
