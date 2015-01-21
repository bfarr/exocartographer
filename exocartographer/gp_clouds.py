import healpy as hp
import george as gg
import numpy as np

def gaussian_cov(times, nside, lambda_spatial, lambda_time, nest=False):
    r"""Returns the covariance matrix (scaled to unit diagonal) for the
    gaussian process model.  The covariance matrix is

    .. math::

      \left \langle x_i x_j \right \rangle = \exp\left[ -\frac{\theta_{ij}^2}{2 \lambda_\theta^2} - \frac{t_{ij}^2}{2 \lambda_t^2} \right],

    where :math:`\theta_{ij}` is the great-circle angular distance
    between the centres of pixels :math:`i` and :math:`j`, and
    :math:`t_{ij}` is the time between the observations of pixels.  

    :param times: Observation times.

    :param nside: The ``nside`` parameter for the healpix pixellated
      map.

    :param lambda_spatial: The spatial correlation length
      (radians---great-circle separation).

    :param lambda_time: The temporal correlation length.

    :param nest: The ordering of the healpix map of pixels.

    :return: A ``(npix*ntime, npix*ntime)`` matrix giving the
      correlation between pixels at different locations and different
      times in our Gaussian process model.  The ordering of pixels is
      consistent with a re-shape into ``(ntime, npix, ntime, npix)``
      four-dimensional array.

    """

    npix = hp.nside2npix(nside)
    n = times.shape[0]*npix

    inds = np.arange(0, npix)
    x,y,z = hp.pix2vec(nside, inds, nest=nest)

    vecs = np.column_stack((x,y,z))

    spatial_angles2 = np.sum(vecs[np.newaxis,:,:]*vecs[:,np.newaxis,:], axis=2)
    print spatial_angles2[(spatial_angles2 > 1.0) | (spatial_angles2 < -1.0)]
    spatial_angles2[spatial_angles2 > 1] = 1.0
    spatial_angles2[spatial_angles2 < -1] = -1.0
    spatial_angles2 = np.square(np.arccos(spatial_angles2))

    time_diff2 = np.square(times[np.newaxis,:] - times[:,np.newaxis])

    scaled_distance2 = time_diff2[:,np.newaxis, :, np.newaxis]/(lambda_time*lambda_time) + spatial_angles2[np.newaxis, :, np.newaxis, :]/(lambda_spatial*lambda_spatial)

    return np.exp(-scaled_distance2/2.0).reshape((n,n))
    
def draw_data_cube(times, nside, mu, sigma, lambda_spatial, lambda_time, nest=False):
    r"""Returns a ``(ntime, npix)`` data array of pixels drawn from the
    quasi-Gaussian process model described in :func:`gaussian_cov`.
    The model is quasi-Gaussian because each pixel's covering fraction
    is derived from a process by 

    .. math::

      f = \frac{1}{1 + \exp(-x)}

    where :math:`x` is drawn from the Gaussian process, and :math:`f`
    is the cloud covering fraction.  

    :param times: The observation times.

    :param nside: The ``nside`` parameter for the healpix pixel map.

    :param mu: The average value of the process generating :math:`x`.

    :param sigma: The standard deviation at zero separation in space
      and time for the process generating :math:`x`.

    :param lambda_spatial: The spatial correlation scale.

    :param lambda_time: The temporal correlation scale.

    :param nest: Ordering of the healpix map.

    :return: A ``(ntime, npix)`` array of covering fractions drawn
      from the GP model.

    """
    cov = sigma*sigma*gaussian_cov(times, nside, lambda_spatial, lambda_time, nest=nest)
    mean = mu*np.ones(cov.shape[0])

    logit = np.random.multivariate_normal(mean=mean, cov=cov)
    data = 1.0/(1.0 + np.exp(-logit))

    return data.reshape((times.shape[0], hp.nside2npix(nside)))
    
def draw_data_cube_george(times, nside, mu, sigma, lambda_spatial, lambda_time, nest=False):
    """Like :func:`draw_data_cube`, but tries to use George (DFM's GP
    model based on Ambikasaran et al's HODLR solvers).  Currently,
    this breaks because there is no HODLR factorisation for the
    Cholesky decomposition needed to *draw* samples from the model.

    """
    
    npix = hp.nside2npix(nside)
    n = times.shape[0]*npix

    inds = np.arange(0, npix)
    x,y,z = hp.pix2vec(nside, inds, nest=nest)

    vecs = np.column_stack((x,y,z))

    lambda_spatial = 2.0*(1-np.cos(lambda_spatial))

    pts = np.column_stack((np.tile(times, (vecs.shape[0],)),
                           np.tile(vecs, (times.shape[0], 1))))

    metric = 1.0/np.array([lambda_time, lambda_spatial, lambda_spatial, lambda_spatial])
    
    kernel = sigma*sigma*gg.kernels.Matern32Kernel(metric, ndim=4)
    gp = gg.GP(kernel, solver=gg.HODLRSolver)

    gp.compute(pts)
    
    logit = gp.sample() + mu

    return 1.0/(1.0 + np.exp(-logit))
