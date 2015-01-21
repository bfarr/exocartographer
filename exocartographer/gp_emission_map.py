import george as gg
import healpy as hp
import numpy as np
import scipy.linalg as sl
import scipy.stats as ss

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

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def cos_theta(self, p):
        p = self.to_params(p)
        return 1.0 / (1.0 + exp(-p['logit_cos_theta']))

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

        normal_vecs = np.column_stack((st*cp, st*sp, ct))
        pixel_vecs = hp.pix2vec(self.nside, np.arange(0, self.npix))

        # dot_products is of shape (ntimes, npix)
        dot_products = np.sum(normal_vecs[:, np.newaxis, :]*pixel_vecs[np.newaxis, :, :], axis=2)

        # Zero out un-observable
        dot_products[dot_products < 0] = 0.0

        log_intensity = np.logaddexp.reduce(np.log(dot_products) + p['log_intensity_map'][np.newaxis, :], axis=1)

        return log_intensity

    def logmapprior(self, p):
        p = self.to_params(p)

        vecs = hp.pix2vec(self.nside, np.arange(0, self.npix))

        dot_prods = np.sum(vecs[np.newaxis, :, :]*vecs[:, np.newaxis, :], axis=2)
        ang_distance = np.arccos(dot_prods)

        sigma = self.sigma(p)
        sp_scale = self.spatial_scale(p)
        mu = p['mu']
        
        cov = sigma*sigma*np.exp(-np.square(ang_distance)/(2.0*sp_scale*sp_scale))

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
        return logpdata(p) + logmapprior(p)
        
