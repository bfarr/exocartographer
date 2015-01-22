import george as gg
import healpy as hp
import numpy as np
import scipy.linalg as sl
import scipy.stats as ss

def logit(x):
    return np.log(x) - np.log(1-x)

def inv_logit(y):
    return 1.0/(1.0 + np.exp(-y))

class EmissionMapPosterior(object):
    def __init__(self, times, intensity, sigma_intensity, nside=4):
        self._times = times
        self._intensity = intensity
        self._sigma_intensity = sigma_intensity
        self._nside = nside
        self._ntimes = times.shape[0]

    @property
    def times(self):
        return self._times

    @property
    def intensity(self):
        return self._intensity

    @property
    def sigma_intensity(self):
        return self._sigma_intensity

    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        return hp.nside2npix(self.nside)

    @property
    def ntimes(self):
        return self._ntimes

    @property
    def dtype(self):
        return np.dtype([('mu', np.float),
                         ('log_sigma', np.float),
                         ('log_spatial_scale', np.float),
                         ('log_period', np.float),
                         ('logit_cos_theta', np.float),
                         ('log_intensity_map', np.float, self.npix)])

    @property
    def nparams(self):
        return 5 + self.npix

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def cos_theta(self, p):
        p = self.to_params(p)
        return inv_logit(p['logit_cos_theta'])

    def period(self, p):
        p = self.to_params(p)
        return np.exp(p['log_period'])
    
    def sigma(self, p):
        p = self.to_params(p)
        return np.exp(p['log_sigma'])

    def spatial_scale(self, p):
        p = self.to_params(p)
        return np.exp(p['log_spatial_scale'])

    def intensity_series(self, p):
        p = self.to_params(p)

        per = self.period(p)        
        cos_theta = self.cos_theta(p)

        phis = 2.0*np.pi*np.fmod(self.times, per)

        ct = cos_theta
        st = np.sqrt(1.0 - ct*ct)
        cp = np.cos(phis)
        sp = np.sin(phis)

        normal_vecs = np.zeros((self.ntimes, 3))
        normal_vecs[:,0] = st*cp
        normal_vecs[:,1] = st*sp
        normal_vecs[:,2] = ct
        pixel_vecs = hp.pix2vec(self.nside, np.arange(0, self.npix))
        pixel_vecs = np.column_stack(pixel_vecs)

        # dot_products is of shape (ntimes, npix)
        dot_products = np.sum(normal_vecs[:, np.newaxis, :]*pixel_vecs[np.newaxis, :, :], axis=2)

        # Zero out un-observable
        dot_products[dot_products < 0] = 0.0

        log_intensity = np.logaddexp.reduce(np.log(dot_products) + p['log_intensity_map'][np.newaxis, :], axis=1)

        return log_intensity

    def map_angular_distances(self):
        vecs = hp.pix2vec(self.nside, np.arange(0, self.npix))
        vecs = np.column_stack(vecs)

        dot_prods = np.sum(vecs[np.newaxis, :, :]*vecs[:, np.newaxis, :], axis=2)
        dot_prods[dot_prods > 1] = 1.0
        dot_prods[dot_prods < -1] = -1.0
        ang_distance = np.arccos(dot_prods)

        return ang_distance
    
    def map_covariance(self, p):
        p = self.to_params(p)

        ang_distance = self.map_angular_distances()

        sigma = self.sigma(p)
        sp_scale = self.spatial_scale(p)
        mu = p['mu']
        
        cov = sigma*sigma*np.exp(-np.square(ang_distance)/(2.0*sp_scale*sp_scale))

        return cov

    def logmapprior(self, p):
        p = self.to_params(p)

        cov = self.map_covariance(p)

        cho_factor, lower = sl.cho_factor(cov)

        log_det = np.sum(np.log(np.diag(cho_factor)))

        n = cov.shape[0]
        
        return -0.5*n*np.log(2.0*np.pi) - log_det \
            - 0.5*np.dot(p['log_intensity_map'] - mu,
                         sl.cho_solve((cho_factor, lower), p['log_intensity_map'] - mu))

    def logpdata(self, p):
        p = self.to_params(p)

        log_ints = self.intensity_series(p)

        return np.sum(ss.norm.logpdf(self.intensity, loc=log_ints, scale=self.sigma_intensity))

    def __call__(self, p):
        lp = self.logpdata(p) + self.logmapprior(p)
        
        return lp
        
