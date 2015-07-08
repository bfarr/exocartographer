import gp_map as gm
import healpy as hp
import numpy as np
import scipy.stats as ss
from util import logit, inv_logit, flat_logit_log_prior

def quaternion_multiply(qa, qb):
    result = np.zeros(qa.shape)

    result[..., 0] = qa[..., 0]*qb[..., 0] - np.sum(qa[..., 1:]*qb[..., 1:], axis=-1)
    result[..., 1] = qa[..., 0]*qb[..., 1] + qa[..., 1]*qb[..., 0] + qa[..., 2]*qb[..., 3] - qa[..., 3]*qb[..., 2]
    result[..., 2] = qa[..., 0]*qb[..., 2] - qa[..., 1]*qb[..., 3] + qa[..., 2]*qb[..., 0] + qa[..., 3]*qb[..., 1]
    result[..., 3] = qa[..., 0]*qb[..., 3] + qa[..., 1]*qb[..., 2] - qa[..., 2]*qb[..., 1] + qa[..., 3]*qb[...,0]

    return result

def rotation_quaternions(axis, angles):
    result = np.zeros((angles.shape[0], 4))
    result[:, 0] = np.cos(angles/2.0)
    result[:, 1:] = np.sin(angles/2.0)[:, np.newaxis]*axis

    return result

def rotate_vector(rqs, v):
    nrs = rqs.shape[0]

    rqs = rqs

    vq = np.zeros((nrs, 4))
    vq[:,1:] = v

    result = quaternion_multiply(rqs, vq)
    rqs[:,1:] *= -1
    result = quaternion_multiply(result, rqs)

    return result[:,1:]

class IlluminationMapPosterior(object):
    def __init__(self, times, intensity, sigma_intensity, nside=4, nside_illum=16):
        assert nside_illum >= nside, 'IlluminationMapPosterior: must have nside_illum >= nside'

        self._times = times
        self._intensity = intensity
        self._sigma_intensity = sigma_intensity
        self._nside = nside
        self._nside_illum = nside_illum

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
    def nside_illum(self):
        return self._nside_illum

    @property
    def ntimes(self):
        return self.times.shape[0]

    @property
    def npix(self):
        return hp.nside2npix(self.nside)

    @property
    def npix_illum(self):
        return hp.nside2npix(self.nside_illum)

    @property
    def dtype(self):
        return np.dtype([('mu', np.float),
                         ('log_sigma', np.float),
                         ('logit_wn_rel_amp', np.float),
                         ('logit_spatial_scale', np.float),
                         ('t0', np.float),
                         ('log_rotation_period', np.float),
                         ('log_orbital_period', np.float),
                         ('logit_cos_inc', np.float),
                         ('logit_cos_obl', np.float),
                         ('logit_phi', np.float)])
    
    @property
    def dtype_map(self):
        return np.dtype([('mu', np.float),
                         ('log_sigma', np.float),
                         ('logit_wn_rel_amp', np.float),
                         ('logit_spatial_scale', np.float),
                         ('t0', np.float),
                         ('log_rotation_period', np.float),
                         ('log_orbital_period', np.float),
                         ('logit_cos_inc', np.float),
                         ('logit_cos_obl', np.float),
                         ('logit_phi', np.float),
                         ('log_albedo_map', np.float, self.npix)])

    @property
    def nparams(self):
        return 10
    @property
    def nparams_map(self):
        return self.nparams + self.npix

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

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def to_params_map(self, p):
        return np.atleast_1d(p).view(self.dtype_map).squeeze()

    def params_map_to_params(self, pm):
        pm = self.to_params_map(pm)
        pp = self.to_params(np.zeros(self.nparams))

        for n in pp.dtype.names:
            pp[n] = pm[n]

        return pp

    def spatial_scale(self, p):
        p = self.to_params(p)

        return inv_logit(p['logit_spatial_scale'],
                         low=self.spatial_scale_low,
                         high=self.spatial_scale_high)

    def wn_rel_amp(self, p):
        p = self.to_params(p)

        return inv_logit(p['logit_wn_rel_amp'],
                         low=self.wn_low,
                         high=self.wn_high)

    def visibility_illumination_matrix_map(self, p):
        return self.visibility_illumination_matrix(self.params_map_to_params(p))
    
    def visibility_illumination_matrix(self, p):
        p = self.to_params(p)
        
        cos_inc = inv_logit(p['logit_cos_inc'])
        sin_inc = np.sqrt(1.0 - cos_inc*cos_inc)
        
        cos_obl = inv_logit(p['logit_cos_obl'], low=-1, high=1)
        sin_obl = np.sqrt(1.0 - cos_obl*cos_obl)

        phi = inv_logit(p['logit_phi'], low=0, high=2*np.pi)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        # In orbit-centred coordinate system
        obs_vec = np.array([sin_inc, 0.0, cos_inc])
        sun_vec = np.array([-1.0, 0.0, 0.0])
        L = np.array([0.0, 0.0, 1.0])

        # Rotate to planet-centred coordinate system
        Rz = np.array([[cos_phi, sin_phi, 0.0],
                       [-sin_phi, cos_phi, 0.0],
                       [0.0, 0.0, 1.0]])
        Ry = np.array([[cos_obl, 0.0, -sin_obl],
                       [0.0, 1.0, 0.0],
                       [sin_obl, 0.0, cos_obl]])
        R = np.dot(Ry, Rz)
        obs_vec = np.dot(R, obs_vec)
        sun_vec = np.dot(R, sun_vec)
        L = np.dot(R, L)

        orbit_angles = 2.0*np.pi/np.exp(p['log_orbital_period'])*(self.times-p['t0'])
        spin_angles = 2.0*np.pi/np.exp(p['log_rotation_period'])*(self.times-p['t0'])

        orbit_rotations = rotation_quaternions(L, orbit_angles)
        spin_rotations = rotation_quaternions(np.array([0.0, 0.0, 1.0]), -spin_angles)

        sun_rotation = quaternion_multiply(spin_rotations, orbit_rotations)
        obs_rotation = spin_rotations

        sun_vectors = rotate_vector(sun_rotation, sun_vec)
        obs_vectors = rotate_vector(obs_rotation, obs_vec)

        pts = hp.pix2vec(self.nside_illum, np.arange(0, self.npix_illum))
        pts = np.column_stack(pts)

        cos_insolation = np.sum(sun_vectors[:,np.newaxis,:]*pts[np.newaxis,:,:], axis=2)
        cos_insolation[cos_insolation > 1] = 1.0
        cos_insolation[cos_insolation < 0] = 0.0

        cos_obs = np.sum(obs_vectors[:,np.newaxis,:]*pts[np.newaxis,:,:], axis=2)
        cos_obs[cos_obs > 1] = 1.0
        cos_obs[cos_obs < 0] = 0.0

        cos_factors = cos_insolation*cos_obs

        area = hp.nside2pixarea(self.nside_illum)

        return area*cos_factors

    def resolved_visibility_illumination_matrix(self, p):
        V = self.visibility_illumination_matrix(p)

        assert self.nside_illum >= self.nside, 'resolution mismatch: nside > nside_illum'

        if self.nside_illum == self.nside:
            return V
        else:
            nside_V = self.nside_illum
            V = np.array(hp.reorder(V, r2n=True))
            while nside_V > self.nside:
                V = V[:,::4] + V[:,1::4] + V[:,2::4] + V[:,3::4]
                nside_V = nside_V / 2
            V = np.array(hp.reorder(V, n2r=True))

            return V

    def visibility_illumination_maps(self, p):
        p = self.to_params_map(p)
        return np.exp(gm.resolve(p['log_albedo_map'], self.nside_illum))*self.visibility_illumination_matrix_map(p)

    def lightcurve_map(self, p):
        return np.log(np.sum(self.visibility_illumination_maps(p), axis=1))

    def log_prior(self, p):
        p = self.to_params(p)

        lp = 0.0

        lp += p['log_sigma'] # Flat prior in sigma

        lp += flat_logit_log_prior(p['logit_wn_rel_amp'],
                                   low=self.wn_low,
                                   high=self.wn_high)

        lp += flat_logit_log_prior(p['logit_spatial_scale'],
                                   low=self.spatial_scale_low,
                                   high=self.spatial_scale_high)

        return lp

    def log_prior_map(self, p):
        p = self.to_params_map(p)

        pp = self.to_params(np.zeros(self.nparams))

        for n in pp.dtype.names:
            pp[n] = p[n]

        return self.log_prior(pp)
    
    def log_pdata_map(self, p):
        p = self.to_params_map(p)

        lightcurve = self.lightcurve(p)

        return np.sum(ss.norm.logpdf(self.intensity, loc=lightcurve, scale=self.sigma_intensity))

    def log_pmap_map(self, p):
        p = self.to_params_map(p)

        sigma = np.exp(p['log_sigma'])
        wn_rel_amp = self.wn_rel_amp(p)
        lambda_spatial = self.spatial_scale(p)

        return gm.map_logprior(p['log_albedo_map'], p['mu'], sigma, wn_rel_amp, lambda_spatial)

    def gp_sigma_matrix(self, p):
        p = self.to_params(p)

        sigma = np.exp(p['log_sigma'])
        wn_rel_amp = self.wn_rel_amp(p)
        lambda_spatial = self.spatial_scale(p)

        Sigma = sigma*sigma*gm.exp_cov(self.nside, wn_rel_amp, lambda_spatial)

        return Sigma

    def data_sigma_matrix(self, inv=False):
        dterm = self.sigma_intensity*self.sigma_intensity*np.exp(2*self.intensity)
        if not inv:
            return np.diag(dterm)
        else:
            return np.diag(1.0/dterm)
        
    
    def gamma_matrix(self, p, V=None):
        p = self.to_params(p)

        Sigma = self.gp_sigma_matrix(p)

        if V is None:
            V = self.resolved_visibility_illumination_matrix(p)

        sigma_matrix = self.data_sigma_matrix()

        M = sigma_matrix + np.dot(V, np.dot(Sigma, V.T))

        MM = np.linalg.solve(M, np.dot(V, Sigma))

        MMM = np.dot(Sigma, np.dot(V.T, MM))

        return Sigma - MMM


    def mbar(self, p, gm=None, V=None):
        p = self.to_params(p)

        if gm is None:
            gm = self.gamma_matrix(p)

        Sigma = self.gp_sigma_matrix(p)

        if V is None:
            V = self.resolved_visibility_illumination_matrix(p)

        dover_sigma = 1.0/(self.sigma_intensity*self.sigma_intensity*np.exp(self.intensity))

        A = np.linalg.solve(Sigma, p['mu']*np.ones(Sigma.shape[0]))
        B = np.dot(V.T, dover_sigma)

        return np.dot(gm, A+B)

    def log_mapmarg_likelihood(self, p):
        p = self.to_params(p)

        V = self.resolved_visibility_illumination_matrix(p)
        gamma = self.gamma_matrix(p, V)
        mbar = self.mbar(p, gamma, V)

        log_mapp = gm.map_logprior(mbar, p['mu'], np.exp(p['log_sigma']), self.wn_rel_amp(p), self.spatial_scale(p))

        map_lc = np.dot(V, mbar)

        residual = np.exp(self.intensity) - map_lc
        sigmas = np.exp(self.sigma_intensity + self.intensity)

        chi2 = np.sum(np.square(residual/sigmas))

        Nd = residual.shape[0]
        Nm = mbar.shape[0]
        
        log_datap = -0.5*Nd*np.log(2.0*np.pi) - np.sum(np.log(sigmas)) - 0.5*chi2

        C = log_datap + log_mapp

        gamma_eigs = np.linalg.eigvalsh(gamma)

        return C + 0.5*Nm*np.log(2.0*np.pi) + 0.5*np.sum(np.log(np.abs(gamma_eigs)))

    def __call__(self, p):
        return self.log_prior(p) + self.log_mapmarg_likelihood(p)

    def log_posterior_map(self, p):
        return self.log_prior_map(p) + self.log_pdata_map(p) + self.log_pmap_map(p)

    def draw_map(self, p):
        V = self.resolved_visibility_illumination_matrix(p)
        gamma = self.gamma_matrix(p, V)
        mbar = self.mbar(p, gamma, V)

        return np.random.multivariate_normal(mbar, gamma)
