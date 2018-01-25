from . import gp_map as gm
import healpy as hp
import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
from .util import logit, inv_logit, flat_logit_log_prior

class EmissionMapPosterior(object):
    def __init__(self, times, intensity, sigma_intensity, nside=4):
        self._times = times
        self._intensity = intensity
        self._sigma_intensity = sigma_intensity
        self._nside = nside
        self._ntimes = times.shape[0]
        self._fix_dict = {}

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

    def fix(self, var, value):
        if var in self.alternate_varnames:
            k,m = self.alternate_varnames[var]
            self._fix_dict[k] = m(value)
        else:
            self._fix_dict[var] = value

    def free(self, var):
        if var in self.alternate_varnames:
            k, m = self.alternate_varnames[var]
            del self._fix_dict[k]
        else:
            del self._fix_dict[var]

    def freeall(self):
        self._fix_dict = {}

    @property
    def varnames(self):
        return ['mu', 'log_sigma', 'logit_wn_rel_amp', 'logit_spatial_scale', 'log_period', 'logit_cos_theta', 'log_intensity_map']
        
    @property
    def alternate_varnames(self):
        return {'sigma': ('log_sigma', np.log),
                'wn_rel_amp': ('logit_wn_rel_amp', lambda wra: logit(wra, self.wn_low, self.wn_high)),
                'spatial_scale': ('logit_spatial_scale', lambda ss: logit(ss, self.spatial_scale_low, self.spatial_scale_high)),
                'period': ('log_period', np.log),
                'cos_theta': ('logit_cos_theta', logit),
                'theta': ('logit_cos_theta', lambda x: logit(cos(x))),
                'intensity_map': ('log_intensity_map', np.log)}

    @property
    def fulldtype(self):
        d = self._fix_dict

        self._fix_dict = {}
        dt = self.dtype
        self._fix_dict = d

        return dt

    @property
    def dtype(self):

        dtl = []

        for n in self.varnames:
            if n not in self._fix_dict:
                if n == 'log_intensity_map':
                    dtl.append((n, np.float, self.npix))
                else:
                    dtl.append((n, np.float))

        return np.dtype(dtl)

    @property
    def nparams(self):

        np = 0
        for n in self.varnames:
            if n not in self._fix_dict:
                if n == 'log_intensity_map':
                    np += self.npix
                else:
                    np += 1
                    
        return np

    @property
    def fullnparams(self):
        d = self._fix_dict

        self._fix_dict = {}
        n = self.nparams
        self._fix_dict = d

        return n

    def to_params(self, p):
        if p.dtype == self.fulldtype:
            return p
        else:
            pp = np.atleast_1d(p).view(self.dtype).squeeze()

            d = self._fix_dict

            self._fix_dict = {}
            pall = np.zeros(self.nparams).view(self.dtype)
            self._fix_dict = d

            for n in pall.dtype.names:
                if n in self._fix_dict:
                    pall[n] = self._fix_dict[n]
                else:
                    pall[n] = pp[n]

            return pall.squeeze()

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

    lightcurve = intensity_series

    def logmapprior(self, p):
        p = self.to_params(p)

        sigma = self.sigma(p)
        wn_rel_amp = inv_logit(p['logit_wn_rel_amp'], low=self.wn_low, high=self.wn_high)
        sp_scale = self.spatial_scale(p)

        return gm.map_logprior_cl(p['log_intensity_map'], p['mu'], sigma, wn_rel_amp, sp_scale)

    def logpdata(self, p):
        p = self.to_params(p)

        log_ints = self.intensity_series(p)

        return np.sum(ss.norm.logpdf(self.intensity, loc=log_ints, scale=self.sigma_intensity))

    def log_prior(self, p):
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
        lp = self.logpdata(p) + self.logmapprior(p) + self.log_prior(p)

        return lp

    # Needed for visualization
    def error_scale(self, p):
        return 1.0

    def hpmap(self, p):
        return np.exp(self.to_params(p)['log_intensity_map'])
