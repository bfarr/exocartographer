import gp_map as gm
import healpy as hp
import numpy as np
import scipy.linalg as sl
import scipy.stats as ss
from util import logit, inv_logit, flat_logit_log_prior
from gp_map import draw_map_cl, map_logprior_cl
from analytic_kernel import viewgeom, kernel

def quaternion_multiply(qa, qb):
    result = np.zeros(np.broadcast(qa, qb).shape)

    result[..., 0] = qa[..., 0]*qb[..., 0] - np.sum(qa[..., 1:]*qb[..., 1:], axis=-1)
    result[..., 1] = qa[..., 0]*qb[..., 1] + qa[..., 1]*qb[..., 0] + qa[..., 2]*qb[..., 3] - qa[..., 3]*qb[..., 2]
    result[..., 2] = qa[..., 0]*qb[..., 2] - qa[..., 1]*qb[..., 3] + qa[..., 2]*qb[..., 0] + qa[..., 3]*qb[..., 1]
    result[..., 3] = qa[..., 0]*qb[..., 3] + qa[..., 1]*qb[..., 2] - qa[..., 2]*qb[..., 1] + qa[..., 3]*qb[...,0]

    return result

def rotation_quaternions(axis, angles):
    angles = np.atleast_1d(angles)
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

        self.fixed_params = None

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
    def lmax(self):
        return self.nside*4 - 1

    @property
    def mmax(self):
        return self.lmax

    @property
    def nalms(self):
        ncomplex = self.mmax * (2 * self.lmax + 1 - self.mmax) / 2 + self.lmax + 1
        return 2*ncomplex

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

    _param_names = ['log_error_scale', 'mu', 'log_sigma', 'logit_wn_rel_amp', 'logit_spatial_scale',
                    'log_rotation_period', 'log_orbital_period',
                    'logit_phi_orb', 'logit_cos_obl', 'logit_phi_rot',
                    'logit_cos_inc']

    @property
    def full_dtype(self):
        return np.dtype([(n, np.float) for n in self._param_names])

    @property
    def dtype(self):
        if self.fixed_params is None:
            return self.full_dtype
        else:
            dt = self.full_dtype
            free_dt = [(param, dt[param]) for param in dt.names
                       if param not in self.fixed_params]

            return np.dtype(free_dt)

    @property
    def dtype_map(self):
        if self.fixed_params is None:
            return self.full_dtype_map

        else:
            dt = self.full_dtype_map
            free_dt = [(param, dt[param]) for param in dt.names
                       if param not in self.fixed_params]

            typel = [(n, np.float) for n in self._param_names
                     if n not in self.fixed_params and 'alms' not in n]

        typel.append(('alms', np.float, self.nalms))
        return np.dtype(typel)

    @property
    def full_dtype_map(self):
        typel = [(n, np.float) for n in self._param_names]
        typel.append(('alms', np.float, self.nalms))
        return np.dtype(typel)

    @property
    def nparams_full(self):
        return 11

    @property
    def nparams(self):
        if self.fixed_params is None:
            return self.nparams_full
        else:
            return self.nparams_full - len(self.fixed_params)

    @property
    def nparams_full_map(self):
        return self.nparams_full + self.nalms

    @property
    def nparams_map(self):
        return self.nparams + self.nalms

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

    def error_scale(self, p):
        return np.exp(self.to_params(p)['log_error_scale'])

    def sigma(self, p):
        p = self.to_params(p)

        return np.exp(p['log_sigma'])

    def wn_rel_amp(self, p):
        p = self.to_params(p)

        return inv_logit(p['logit_wn_rel_amp'], self.wn_low, self.wn_high)

    def spatial_scale(self, p):
        p = self.to_params(p)

        return inv_logit(p['logit_spatial_scale'], self.spatial_scale_low, self.spatial_scale_high)

    def rotation_period(self, p):
        return np.exp(self.to_params(p)['log_rotation_period'])

    def orbital_period(self, p):
        return np.exp(self.to_params(p)['log_orbital_period'])

    def phi_orb(self, p):
        return inv_logit(self.to_params(p)['logit_phi_orb'], low=0, high=2*np.pi)

    def cos_obl(self, p):
        return inv_logit(self.to_params(p)['logit_cos_obl'], low=0, high=1)

    def obl(self, p):
        return np.arccos(self.cos_obl(p))

    def phi_rot(self, p):
        return inv_logit(self.to_params(p)['logit_phi_rot'], low=0, high=2*np.pi)

    def cos_inc(self, p):
        return inv_logit(self.to_params(p)['logit_cos_inc'], low=0, high=1)

    def inc(self, p):
        return np.arccos(self.cos_inc(p))

    def set_params(self, p, dict):
        p = np.atleast_1d(p).view(self.dtype)

        logit_names = {'wn_rel_amp': (self.wn_low, self.wn_high),
                       'spatial_scale': (self.spatial_scale_low, self.spatial_scale_high),
                       'phi_orb': (0, 2*np.pi),
                       'cos_obl': (0, 1),
                       'phi_rot': (0, 2*np.pi),
                       'cos_inc': (0, 1)}
        log_names = set(['err_scale', 'sigma', 'rotation_period', 'orbital_period'])

        for n, x in dict.items():
            if n in p.dtype.names:
                p[n] = x
            elif n in logit_names:
                l,h = logit_names[n]
                p['logit_' + n] = logit(x, l, h)
            elif n in log_names:
                p['log_' + n] = np.log(x)

        return p

    def unfix_params(self):
        self.fixed_params = None

    def fix_params(self, fixed_params):
        self.fixed_params = fixed_params

    def to_params(self, p):
        if isinstance(p, np.ndarray):
            if p.dtype == self.full_dtype or p.dtype == self.full_dtype_map:
                return p.squeeze()
            else:
                if p.dtype == self.dtype:
                    # Extend the array with the fixed parameters
                    pp = np.empty(p.shape, dtype=self.full_dtype)
                    for n in p.dtype.names:
                        pp[n] = p[n]
                    for n in self.fixed_params.keys():
                        pp[n] = self.fixed_params[n]
                    return pp.squeeze()
                elif p.dtype == self.dtype_map:
                    pp = np.empty(p.shape, dtype=self.full_dtype_map)
                    for n in p.dtype.names:
                        pp[n] = p[n]
                    for n in self.fixed_params.keys():
                        pp[n] = self.fixed_params[n]
                    return pp.squeeze()
                else:
                    if p.shape[-1] == self.nparams:
                        return self.to_params(p.view(self.dtype).squeeze())
                    elif p.shape[-1] == self.nparams_map:
                        return self.to_params(p.view(self.dtype_map).squeeze())
                    else:
                        print(p.shape[-1], self.nparams_map)
                        raise ValueError("to_params: bad parameter dimension")
        else:
            p = np.atleast_1d(p)
            return self.to_params(p)

    def add_map_to_full_params(self, p, map):
        return self._add_map(p, map)

    def params_map_to_params(self, pm, include_fixed_params=False):
        pm = self.to_params(pm)
        pp = self.to_params(np.zeros(self.nparams))

        for n in pp.dtype.names:
            pp[n] = pm[n]

        if not include_fixed_params:
            unfixed_params = [p for p in pp.dtype.names if p not in self.fixed_params]
            pp = pp[unfixed_params]

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

    # This is the key function that produces the "kernel" of
    # illumination that is integrated against the pixel map.  Here is
    # the logic:
    #
    # The kernel is composed of a product of cosines: the illumination
    # is proportional to n*n_s, where n is the pixel normal and n_s is
    # the vector to the star.  The visibility is proportional to
    # n*n_o, where n_o is the vector to the observer.  The
    # contribution of any pixel of value p to the lightcurve is
    # therefore p*(n*n_s)*(n*n_o) if both n*n_s and n*n_o are > 0, and
    # zero otherwise.  So, we need to evaluate these dot products.
    #
    # Fix a coordinate system in which to evaluate these dot products
    # as follows:
    #
    # The orbit is in the x-y plane (so L ~ z), with the x-axis
    # pointing along superior conjunction.  The observer is therefore
    # in the x-z plane, and has an inclination angle i in [0, pi/2] to
    # the orbital plane.  So n_o = (-sin(i), 0, cos(i)).
    #
    # The planet star vector, n_s, is given by n_s = (-sin(xi),
    # -cos(xi), 0), where xi is the orbital phase.  If the orbit has
    # phase xi0 at t = 0, then n_s = R_z(2*pi/Porb*t + xi0)*(-1,0,0).
    #
    # For the normal vector to the planet, we must describe the series
    # of rotations that maps the orbital coordinate system into the
    # planet-centred coordinate system.  Imagine that the planet spin
    # axis is at first aligned with the z-axis.  We apply R_y(obl),
    # where obl is the obliquity angle in [0, pi/2], and then
    # R_z(phi_rot), where phi_rot is the azimuthal angle of the
    # planet's spin axis in [0, 2*pi].  Now the planet's spin axis
    # points to S = (cos(phi_rot)*sin(obl), sin(phi_rot)*sin(obl),
    # cos(obl)).  At time t, the planet's normals are given by n(t) =
    # R_S(2*pi/Prot*t)*n(0), where n(0) = R_z(phi_rot)*R_y(obl)*n,
    # with n the body-centred normals to the pixels.  We can now
    # evaluate dot products in the fixed orbital frame.
    #
    # In principle, we are done, but there is an efficiency
    # consideration.  For each time at which we have an observation,
    # we need to perform rotations on the various vectors in the
    # problem.  There are, in general, a large number of n vectors (we
    # have many pixels covering the planet's surface), but only one
    # n_o and n_s vector.  It will be more efficient to apply
    # rotations only to the n_o and n_s vectors instead of rotating
    # the many, many, n vectors.  (This is equivalent to performing
    # the dot products in the planet-centred frame; we could have done
    # that from the beginning, but it is easier to describe the
    # geometry in the orbit-centred frame.)  We can accomplish this
    # via a trick: when vectors in a dot product are rotated, the
    # rotations can be moved from one vector to the other:
    #
    # (R_A*a)*(R_B*b) = < R_A * a | R_B * b > = < R_B^-1 R_A * a | b > = < a | R_A^-1 R_B b >
    #
    # So, instead of rotating the normal vectors to the planet pixels by
    #
    # n(t) = R_S(omega_rot(t))*R_z(phi_rot)*R_y(obl)*n
    #
    # We can just rotate the vectors that are inner-producted with
    # these vectors by
    #
    # R_inv = R_y(-obl)*R_z(-phi_rot)*R_S(-omega_rot)
    #
    # The code below should be implementing this.
    def visibility_illumination_matrix(self, p):
        p = self.to_params(p)

        phi_orb = inv_logit(p['logit_phi_orb'], 0.0, 2.0*np.pi)
        cos_phi_orb = np.cos(phi_orb)
        sin_phi_orb = np.sin(phi_orb)

        omega_orb = 2.0*np.pi/np.exp(p['log_orbital_period'])

        no = self._observer_normal_orbit_coords(p)
        ns = np.array([-cos_phi_orb, -sin_phi_orb, 0.0])

        orb_quats = rotation_quaternions(np.array([0.0, 0.0, 1.0]), -omega_orb*self.times)

        to_body_frame_quats = self._body_frame_quaternions(p, self.times)
        star_to_bf_quats = quaternion_multiply(to_body_frame_quats,orb_quats)

        nos = rotate_vector(to_body_frame_quats, no)
        nss = rotate_vector(star_to_bf_quats, ns)

        pts = hp.pix2vec(self.nside_illum, np.arange(0, self.npix_illum))
        pts = np.column_stack(pts)

        cos_insolation = np.sum(nss[:,np.newaxis,:]*pts[np.newaxis,:,:], axis=2)
        cos_insolation[cos_insolation > 1] = 1.0
        cos_insolation[cos_insolation < 0] = 0.0

        cos_obs = np.sum(nos[:,np.newaxis,:]*pts[np.newaxis,:,:], axis=2)
        cos_obs[cos_obs > 1] = 1.0
        cos_obs[cos_obs < 0] = 0.0

        cos_factors = cos_insolation*cos_obs

        area = hp.nside2pixarea(self.nside_illum)

        return area * cos_factors/np.pi

    #TODO: rename phi_rot to obl_orientation
    def analytic_visibility_illumination_matrix(self, p):
        omega_rot = 2.0*np.pi/self.rotation_period(p)
        obl = self.obl(p)
        obl_orientation = self.phi_rot(p)
        omega_orb = 2.0*np.pi/self.orbital_period(p)
        phi_orb = self.phi_orb(p)
        inc = self.inc(p)

        lon, lat = hp.pix2ang(self.nside_illum, np.arange(0, self.npix_illum))
        sinth = np.sin(lat)
        costh = np.cos(lat)
        sinph = np.sin(lon)
        cosph = np.cos(lon)

        #Getting sub_obs and sub_st trig values
        all_trigs = viewgeom(self.times, omega_rot, omega_orb, obl, inc, obl_orientation, phi_orb)

        area = hp.nside2pixarea(self.nside_illum)

        return area * kernel(sinth, costh, sinph, cosph, all_trigs).T

    def _body_frame_quaternions(self, p, times):
        p = self.to_params(p)

        times = np.atleast_1d(times)

        cos_obl = inv_logit(p['logit_cos_obl'], 0.0, 1.0)
        sin_obl = np.sqrt(1.0 - cos_obl*cos_obl)
        obl = np.arccos(cos_obl)

        phi_rot = inv_logit(p['logit_phi_rot'], 0.0, 2.0*np.pi)
        cos_phi_rot = np.cos(phi_rot)
        sin_phi_rot = np.sin(phi_rot)

        omega_rot = 2.0*np.pi/np.exp(p['log_rotation_period'])

        S = np.array([cos_phi_rot*sin_obl, sin_phi_rot*sin_obl, cos_obl])

        spin_rot_quat = quaternion_multiply(rotation_quaternions(np.array([0.0, 1.0, 0.0]), -obl), \
                                            rotation_quaternions(np.array([0.0, 0.0, 1.0]), -phi_rot))

        rot_quats = rotation_quaternions(S, -omega_rot*times)

        return quaternion_multiply(spin_rot_quat,rot_quats)

    def _observer_normal_orbit_coords(self, p):
        p = self.to_params(p)

        cos_inc = inv_logit(p['logit_cos_inc'], 0.0, 1.0)
        sin_inc = np.sqrt(1.0 - cos_inc*cos_inc)

        return np.array([-sin_inc, 0.0, cos_inc])

    def resolved_visibility_illumination_matrix(self, p):
        V = self.visibility_illumination_matrix(p)
        #V = self.analytic_visibility_illumination_matrix(p)

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

        logdt = np.log(self.times[1] - self.times[0])
        logT = np.log(self.times[-1] - self.times[0])

        lp += self.log_period_prior(p['log_rotation_period'])
        lp += self.log_period_prior(p['log_orbital_period'])

        lp += flat_logit_log_prior(p['logit_phi_orb'], low=0, high=2*np.pi)
        lp += flat_logit_log_prior(p['logit_cos_obl'], low=0, high=1)
        lp += flat_logit_log_prior(p['logit_phi_rot'], low=0, high=2*np.pi)
        lp += flat_logit_log_prior(p['logit_cos_inc'], low=0, high=1)

        return lp

    def log_period_prior(self, logP):
        logdt = np.log(self.times[1] - self.times[0])
        logT = np.log(self.times[-1] - self.times[0])

        if logP <= logdt:
            lp = -logdt
        if logdt < logP <= logT:
            lp = -logP
        if logP > logT:
            lp = logT - 2.*logP
        lp -= np.log(2. + logT - logdt)

        return lp

    def gp_sigma_matrix(self, p):
        p = self.to_params(p)

        sigma = np.exp(p['log_sigma'])
        wn_rel_amp = self.wn_rel_amp(p)
        lambda_spatial = self.spatial_scale(p)

        Sigma = sigma*sigma*gm.exp_cov(self.nside, wn_rel_amp, lambda_spatial)

        return Sigma

    def data_sigma_matrix(self, inv=False):
        if not inv:
            return np.diag(np.square(self.sigma_intensity))
        else:
            return np.diag(1.0/np.square(self.sigma_intensity))

    def hpmap(self, p):
        p = self.to_params(p)
        complex_alms = p['alms'][:self.nalms/2] + 1j*p['alms'][self.nalms/2:]
        return hp.alm2map(complex_alms, self.nside, lmax=self.lmax, mmax=self.mmax, verbose=False)

    def lightcurve(self, p):
        p = self.to_params(p)

        V = self.resolved_visibility_illumination_matrix(p)
        hpmap = self.hpmap(p)

        map_lc = np.dot(V, hpmap)

        return map_lc

    def loglikelihood(self, p):
        p = self.to_params(p)

        errsc = np.exp(p['log_error_scale'])

        V = self.resolved_visibility_illumination_matrix(p)

        hpmap = self.hpmap(p)

        log_mapp = map_logprior_cl(hpmap, p['mu'], np.exp(p['log_sigma']), self.wn_rel_amp(p), self.spatial_scale(p))

        map_lc = np.dot(V, hpmap)

        residual = self.intensity - map_lc
        sigmas = self.sigma_intensity

        chi2 = np.sum(np.square(residual/(errsc*sigmas)))

        Nd = residual.shape[0]

        log_datap = -0.5*Nd*np.log(2.0*np.pi) - np.sum(np.log(errsc*sigmas)) - 0.5*chi2

        return log_datap + log_mapp

    def __call__(self, p):
        return self.log_prior(p) + self.loglikelihood(p)

    def draw_map(self, p):
        return draw_map_cl(self.nside, p['mu'], np.exp(p['log_sigma']), self.wn_rel_amp(p), self.spatial_scale(p))

    def sub_observer_latlong(self, p, times):
        no = self._observer_normal_orbit_coords(p)

        body_quats = self._body_frame_quaternions(p, times)

        no_body = rotate_vector(body_quats, no)

        colats = np.arccos(no_body[:,2])
        longs = np.arctan2(no_body[:,1], no_body[:,0])

        longs[longs < 0] += 2*np.pi

        return np.pi/2.0-colats, longs
