import numpy as np

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
