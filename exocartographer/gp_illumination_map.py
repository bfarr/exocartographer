import gp_map as gm
import healpy as hp
import numpy as np
import scipy.stats as ss

def logit(x, low=0, high=1):
    return np.log(x-low) - np.log(high-x)

def inv_logit(y, low=0, high=1):
    ey = np.exp(y)
    return low/(1.0 + ey) + high/(1.0 + 1.0/ey)

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
    def __init__(self, times, intensity, sigma_intensity, nside=4):
        self._times = times
        self._intensity = intensity
        self._sigma_intensity = sigma_intensity
        self._nside = nside

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
    def ntimes(self):
        return self.times.shape[0]

    @property
    def npix(self):
        return hp.nside2npix(self.nside)

    @property
    def dtype(self):
        return np.dtype([('mu', np.float),
                         ('log_sigma', np.float),
                         ('log_wn_rel_amp', np.float),
                         ('log_spatial_scale', np.float),
                         ('t0', np.float),
                         ('log_rotation_period', np.float),
                         ('log_orbital_period', np.float),
                         ('logit_cos_inc', np.float),
                         ('logit_cos_obl', np.float),
                         ('logit_phi', np.float),
                         ('log_albedo_map', np.float, self.npix)])

    @property
    def nparams(self):
        return self.npix + 10

    def to_params(self, p):
        return np.atleast_1d(p).view(self.dtype).squeeze()

    def lightcurve(self, p):
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
                       [sin_obl, 0.0, 1.0]])
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

        pts = hp.pix2vec(self.nside, np.arange(0, self.npix))
        pts = np.column_stack(pts)

        cos_insolation = np.sum(sun_vectors[:,np.newaxis,:]*pts[np.newaxis,:,:], axis=2)
        cos_insolation[cos_insolation > 1] = 1.0
        cos_insolation[cos_insolation < 0] = 0.0

        cos_obs = np.sum(obs_vectors[:,np.newaxis,:]*pts[np.newaxis,:,:], axis=2)
        cos_obs[cos_obs > 1] = 1.0
        cos_obs[cos_obs < 0] = 0.0

        cos_factors = cos_insolation*cos_obs

        return np.logaddexp.reduce(p['log_albedo_map'] + np.log(cos_factors), axis=1)

    def log_prior(self, p):
        p = self.to_params(p)

        lp = 0.0
        
        log_theta_pix = np.log(hp.nside2resol(self.nside))
        log_pi = np.log(np.pi)

        if p['log_spatial_scale'] < log_theta_pix:
            lp -= np.square(p['log_spatial_scale'] - log_theta_pix)
        elif p['log_spatial_scale'] > log_pi:
            lp -= np.square(p['log_spatial_scale'] - log_pi)

        log_small = np.log(1e-2)
        log_big = np.log(1e2)
        if p['log_wn_rel_amp'] < log_small:
            lp -= np.square(p['log_wn_rel_amp'] - log_small)
        elif p['log_wn_rel_amp'] > log_big:
            lp -= np.square(p['log_wn_rel_amp'] - log_big)

        return lp        
    
    def log_pdata(self, p):
        p = self.to_params(p)

        lightcurve = self.lightcurve(p)

        return np.sum(ss.norm.logpdf(self.intensity, loc=lightcurve, scale=self.sigma_intensity))

    def log_pmap(self, p):
        p = self.to_params(p)

        sigma = np.exp(p['log_sigma'])
        wn_rel_amp = np.exp(p['log_wn_rel_amp'])
        lambda_spatial = np.exp(p['log_spatial_scale'])

        return gm.map_logprior(p['log_albedo_map'], p['mu'], sigma, wn_rel_amp, lambda_spatial)

    def __call__(self, p):
        return self.log_pdata(p) + self.log_pmap(p) + self.log_prior(p)
