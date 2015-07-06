"""Utilities to implement Gaussian process priors on healpix maps.

"""

import healpy as hp
import numpy as np
import scipy.linalg as sl
import warnings

def exp_cov(nside, wn_rel_amp, lambda_angular, nest=False):
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

    cov = np.exp(-thetas*thetas/(2.0*lambda_angular*lambda_angular))
    cov[np.diag_indices(hp.nside2npix(nside))] = 1.0
    cov[np.diag_indices(hp.nside2npix(nside))] += wn_rel_amp
    cov /= (1.0 + wn_rel_amp)

    return cov

def map_logprior(hpmap, mu, sigma, wn_rel_amp, lambda_angular, nest=False):
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
    
    cov = sigma*sigma*exp_cov(nside, wn_rel_amp, lambda_angular, nest=nest)

    x = hpmap - mu

    try:
        chof, low = sl.cho_factor(cov)
    except sl.LinAlgError:
        warnings.warn('linear algebra error in gp_map Cholesky factorization; retuning -inf')
        return np.NINF

    logdet = np.sum(np.log(np.diag(chof)))

    return -0.5*n*np.log(2*np.pi) - logdet - 0.5*np.dot(x, sl.cho_solve((chof, low), x))

def draw_map(nside, mu, sigma, wn_rel_amp, lambda_spatial, nest=False):
    """Returns a map sampled from the Gaussian process with the given
    parameters.

    :param mu: The mean of the GP (constant on the sphere).

    :param sigma: The standard deviation of the GP at zero angular
      separation.

    :param lambda_spatial: The angular correlation length of the
      process (radians).

    :param nest: Healpix map ordering.

    :return: Healpix map drawn from the GP with the given parameters.

    """

    n = hp.nside2npix(nside)

    cov = sigma*sigma*exp_cov(nside, wn_rel_amp, lambda_spatial, nest=nest)
    mean = mu*np.ones(n)

    return np.random.multivariate_normal(mean, cov)
