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

      C_{ij} = \left \langle p_i p_j \right\rangle = \exp\left( -\frac{1}{2} \left(\frac{\theta_{ij}}{\lambda}\right)^2 \right)

    where :math:`\theta_{ij}` is the great-circle angle between the
    :math:`i` and :math:`j` pixel centres.  

    :param nside: Healpix ``nside`` parameter.

    :param wn_rel_amp: The relative amplitude of white-noise added to
      the covariance matrix.  In other words, the pixel variance is
      always 1, but ``wn_rel_amp`` fraction of it comes from white
      noise, and ``1-wn_rel_amp`` of it from the correlated noise with
      the above angular correlation function.

    :param lambda_angular: The angular correlation length (radians).

    :param nest: Ordering flag for the healpix map.

    """

    vecs = hp.pix2vec(nside, np.arange(0, hp.nside2npix(nside)), nest=nest)
    vecs = np.column_stack(vecs)

    dot_prods = np.sum(vecs[np.newaxis, :, :]*vecs[:, np.newaxis, :], axis=2)
    dot_prods[dot_prods > 1] = 1.0
    dot_prods[dot_prods < -1] = -1.0
    
    thetas = np.arccos(dot_prods)

    cov = np.exp(-0.5*thetas*thetas/(lambda_angular*lambda_angular))
    cov *= 1.0 - wn_rel_amp
    cov[np.diag_indices(hp.nside2npix(nside))] += wn_rel_amp
    cov /= (1.0 + wn_rel_amp)

    return cov

def exp_cov_cl(nside, wn_rel_amp, lambda_angular):
    r"""Returns the equivalent :math:`C_l` angular power spectrum to the
    covariance returned by :func:`exp_cov`.  The returned array is
    normalised by the Healpy convention.

    """

    lscale = np.pi/lambda_angular
    lmax = 3*nside
    ls = np.arange(0, lmax, dtype=np.int)

    cl_corr_unnorm = np.exp(-0.5*np.square(ls/lscale))
    cl_corr_norm = cl_corr_unnorm*4.0*np.pi/np.sum((2*ls+1)*cl_corr_unnorm)

    cl_white_norm = 4.0*np.pi/(lmax+1.0)/(lmax+1.0)

    return (1.0-wn_rel_amp)*cl_corr_norm + wn_rel_amp*cl_white_norm

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

def map_logprior_cl(map, mu, sigma, wn_rel_amp, lambda_angular):
    r"""Returns the GP prior on maps, consistent with :func:`map_logprior`,
    but computed much more quickly.

    """
    npix = map.shape[0]
    nside = hp.npix2nside(npix)
    
    lmax = 3*nside
    ls = np.arange(0, lmax, dtype=np.int)

    cl = exp_cov_cl(nside, wn_rel_amp, lambda_angular)

    map0 = map-mu
    alm_map0 = hp.map2alm(map0)
    alm_map0_white = hp.almxfl(alm_map0, 1.0/np.sqrt(cl))
    
    

    return -0.5*npix*np.log(2.0*np.pi) - npix*np.log(sigma) - 0.5*np.sum(np.square(white_map/sigma))

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

def draw_map_cl(nside, mu, sigma, wn_rel_amp, lambda_spatial):
    r"""Returns a map statistically equivalent to :func:`draw_map`, but is
    much faster.

    """
    cl = exp_cov_cl(nside, wn_rel_amp, lambda_spatial)

    map = hp.synfast(cl, nside, sigma=0)

    return mu + sigma*map

def resolve(pix, new_nside, nest=False):
    nside = hp.npix2nside(pix.shape[0])

    if not nest:
        rpix = resolve(hp.reorder(pix, r2n=True), new_nside, nest=True)
        return hp.reorder(rpix, n2r=True)
    else:
        if nside == new_nside:
            return pix
        elif nside < new_nside:
            # Interpolate up one resolution
            nnew = hp.nside2npix(nside*2)
            inew = np.arange(0, nnew, dtype=np.int)
            
            new_pix = np.zeros(nnew)

            new_pix = pix[inew//4]

            return resolve(new_pix, new_nside, nest=nest)
        else:
            # Intepolate down one resolution
            nnew = hp.nside2npix(nside/2)
            inew = np.arange(0, nnew, dtype=np.int)

            new_pix = np.zeros(nnew)

            new_pix = 0.25*(pix[::4] + pix[1::4] + pix[2::4] + pix[3::4])

            return resolve(new_pix, new_nside, nest=nest)
            
