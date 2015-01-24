"""Utilities to implement Gaussian process priors on healpix maps.

"""

import healpy as hp
import numpy as np
import scipy.linalg as sl

def exp_cov(nside, lambda_angular, nest=False):
    """Returns a covariance function for a healpix map with ``nside``
    using

    .. math::

      C_{ij} = \left \langle p_i p_j \right\rangle = \exp\left( - \frac{\theta_{ij}}{\lambda} \right)

    where :math:`\theta_{ij}` is the great-circle angle between the
    :math:`i` and :math:`j` pixel centres.  

    :param nside: Healpix ``nside`` parameter.

    :param lambda_angular: The angular correlation length (radians).

    :param nest: Ordering flag for the healpix map.

    """

    vecs = hp.pix2vec(nside, np.arange(0, hp.nside2npix(nside)), nest=nest)
    vecs = np.column_stack(vecs)

    dot_prods = np.sum(vecs[np.newaxis, :, :]*vecs[:, np.newaxis, :], axis=2)
    dot_prods[dot_prods > 1] = 1.0
    dot_prods[dot_prods < -1] = -1.0
    
    thetas = np.arccos(dot_prods)

    return np.exp(-thetas / lambda_angular)

def map_logprior(hpmap, mu, sigma, lambda_angular, nest=False):
    """Returns the GP prior on the map with exponential covariance
    function.

    :param hpmap: Healpix map on which the prior is to be evaluated.

    :param mu: Mean of the GP

    :param sigma: Standard deviation at zero angular separation.

    :param lambda_angular: Angular correlation length.

    :param nest: The ordering of the healpix map.

    """

    nside = hp.npix2nside(hpmap.shape[0])
    n = hpmap.shape[0]
    
    cov = sigma*sigma*exp_cov(nside, lambda_angular, nest=nest)

    x = hpmap - mu

    try:
        cho, lower = sl.cho_factor(cov)
    except sl.LinAlgError:
        return np.NINF

    logdet = np.sum(np.log(np.diag(cho)))

    return -0.5*n*np.log(2.0*np.pi) - logdet - 0.5*np.dot(x, sl.cho_solve((cho, lower), x))
