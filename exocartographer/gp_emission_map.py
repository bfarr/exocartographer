import gp_map as gm
import healpy as hp
import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
from util import logit, inv_logit, flat_logit_log_prior

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
    def wn_low(self):
        return 0.01

    @property
    def wn_high(self):
        return 100.0

    @property
    def spatial_scale_low(self):
        return hp.nside2resol(self.nside)/3.0

    @property
    def spatial_scale_high(self):
        return 3.0*np.pi

    @property
    def ntimes(self):
        return self._ntimes

    @property
    def dtype(self):
        return np.dtype([('mu', np.float),
                         ('log_sigma', np.float),
                         ('logit_wn_rel_amp', np.float),
                         ('logit_spatial_scale', np.float),
                         ('log_period', np.float),
                         ('logit_cos_theta', np.float),
                         ('log_intensity_map', np.float, self.npix)])

    @property
    def nparams(self):
        return 6 + self.npix

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

        return inv_logit(p['logit_spatial_scale'],
                         low=self.spatial_scale_low,
                         high=self.spatial_scale_high)

    def wn_rel_amp(self, p):
        p = self.to_params(p)
        return inv_logit(p['logit_wn_rel_amp'], low=self.wn_low, high=self.wn_high)

    def visibility_series(self, p):
        p = self.to_params(p)

        per = self.period(p)
        cos_theta = self.cos_theta(p)

        phis = 2.0*np.pi/per*self.times

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

        return dot_products

    def spatial_intensity_series(self, p):
        p = self.to_params(p)

        dot_products = self.visibility_series(p)

        return np.log(dot_products) + p['log_intensity_map'][np.newaxis, :]

    def intensity_series(self, p):
        return np.logaddexp.reduce(self.spatial_intensity_series(p), axis=1)

    def logmapprior(self, p):
        p = self.to_params(p)

        sigma = self.sigma(p)
        wn_rel_amp = inv_logit(p['logit_wn_rel_amp'], low=self.wn_low, high=self.wn_high)
        sp_scale = self.spatial_scale(p)

        return gm.map_logprior(p['log_intensity_map'], p['mu'], sigma, wn_rel_amp, sp_scale)

    def logpdata(self, p):
        p = self.to_params(p)

        log_ints = self.intensity_series(p)

        return np.sum(ss.norm.logpdf(self.intensity, loc=log_ints, scale=self.sigma_intensity))

    def logprior(self, p):
        p = self.to_params(p)

        lp = 0.0

        lp += flat_logit_log_prior(p['logit_wn_rel_amp'],
                                   low=self.wn_low,
                                   high=self.wn_high)

        lp += flat_logit_log_prior(p['logit_spatial_scale'],
                                   low=self.spatial_scale_low,
                                   high=self.spatial_scale_high)

        return lp


    def __call__(self, p):
        lp = self.logpdata(p) + self.logmapprior(p) + self.logpdata(p)

        return lp
