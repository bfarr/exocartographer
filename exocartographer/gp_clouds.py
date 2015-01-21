import healpy as hp
import george as gg
import numpy as np

def gaussian_cov(times, nside, lambda_spatial, lambda_time, nest=False):
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
    cov = sigma*sigma*gaussian_cov(times, nside, lambda_spatial, lambda_time, nest=nest)
    mean = mu*np.ones(cov.shape[0])

    logit = np.random.multivariate_normal(mean=mean, cov=cov)
    data = 1.0/(1.0 + np.exp(-logit))

    return data.reshape((times.shape[0], hp.nside2npix(nside)))
    
def draw_data_cube_george(times, nside, mu, sigma, lambda_spatial, lambda_time, nest=False):
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
