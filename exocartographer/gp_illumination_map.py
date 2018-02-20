from __future__ import division

import numpy as np
import healpy as hp

from . import gp_map as gm

from .gp_map import draw_map_cl, map_logprior_cl
from .util import logit, inv_logit, flat_logit_log_prior

from .analytic_kernel import viewgeom, kernel


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
    """A posterior class for mapping surfaces from a reflectance time series"

    :param times:
        Array of times of photometric measurements.

    :param reflectance:
        Array of photometric measurements at corresponding `times`.

    :param sigma_reflectance:
        Single value or array of 1-sigma uncertainties on reflectance measurements.

    :param nside: (optional)
        Resolution of HEALPix surface map.  Has to be a power of 2.  The number
        of pixels will be :math:`12 N_\mathrm{side}^2`.
        (default: ``4``)

    :param nside_illum: (optional):
        Resolution of HEALPix illumination map, i.e., the "kernel" of
        illumination that is integrated against the pixel map.
        (default: ``16``)

    :param map_parameterization: (optional):
        Parameterization of surface map to use.  `pix` will parameterize the map with
        values of each pixel; `alm` will parameterize the map in spherical harmonic coefficients.
        (default: ``pix``)

    """
    def __init__(self, times, reflectance, sigma_reflectance, nside=4, nside_illum=16, map_parameterization='pix'):
        assert nside_illum >= nside, 'IlluminationMapPosterior: must have nside_illum >= nside'

        self._map_parameterization = map_parameterization

        self._times = np.array(times)
        self._reflectance = np.array(reflectance)
        self._sigma_reflectance = sigma_reflectance * np.ones_like(times)
        self._nside = nside
        self._nside_illum = nside_illum

        self._fixed_params = {}

    _param_names = ['log_error_scale', 'mu', 'log_sigma', 'logit_wn_rel_amp', 'logit_spatial_scale',
                    'log_rotation_period', 'log_orbital_period',
                    'logit_phi_orb', 'logit_cos_obl', 'logit_obl_orientation',
                    'logit_cos_inc']

    @property
    def times(self):
        return self._times

    @property
    def ntimes(self):
        return self.times.shape[0]

    @property
    def reflectance(self):
        return self._reflectance

    @property
    def sigma_reflectance(self):
        return self._sigma_reflectance

    @property
    def map_parameterization(self):
        return self._map_parameterization

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
        return int(2*ncomplex)

    @property
    def npix(self):
        return hp.nside2npix(self.nside)

    @property
    def nmap_params(self):
        if self.map_parameterization == 'pix':
            return self.npix
        elif self.map_parameterization == 'alm':
            return self.nalms
        else:
            raise RuntimeError("Unrecognized map parameterization {}".format(self.map_parameterization))

    @property
    def nside_illum(self):
        return self._nside_illum

    @property
    def npix_illum(self):
        return hp.nside2npix(self.nside_illum)

    @property
    def fixed_params(self):
        return self._fixed_params

    @property
    def full_dtype(self):
        return np.dtype([(n, np.float) for n in self._param_names])

    @property
    def dtype(self):
        dt = self.full_dtype
        free_dt = [(param, dt[param]) for param in dt.names
                   if param not in self.fixed_params]

        return np.dtype(free_dt)

    @property
    def dtype_map(self):
        typel = [(param, self.dtype[param]) for param in self.dtype.names]
        typel.append(('map', np.float, self.nmap_params))
        return np.dtype(typel)

    @property
    def full_dtype_map(self):
        typel = [(n, self.full_dtype[n]) for n in self.full_dtype.names]
        typel.append(('map', np.float, self.nmap_params))
        return np.dtype(typel)

    @property
    def nparams_full(self):
        return len(self.full_dtype)

    @property
    def nparams(self):
        return len(self.dtype)

    @property
    def nparams_full_map(self):
        return self.nparams_full + self.nmap_params

    @property
    def nparams_map(self):
        return self.nparams + self.nmap_params

    @property
    def wn_low(self):
        return 0.01

    @property
    def wn_high(self):
        return 0.99

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

    def obl_orientation(self, p):
        return inv_logit(self.to_params(p)['logit_obl_orientation'], low=0, high=2*np.pi)

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
                       'obl_orientation': (0, 2*np.pi),
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

    def fix_params(self, params):
        """
        Fix parameters to the specified values and remove them from the Posterior's `dtype`.

        Args:
            params (dict): A dictionary of parameters to fix, and the
                values to fix them to.

        """
        self._fixed_params.update(params)

    def unfix_params(self, params=None):
        """
        Let fixed parameters vary.

        Args:
            params (iterable): A list of parameters to unfix.  If ``None``, all
            parameters will be allowed to vary.
            (default: ``None``)

        """
        if params is None:
            self._fixed_params = {}
        else:
            for p in params:
                try:
                    self._fixed_params.pop(p)
                except KeyError:
                    continue

    def to_params(self, p):
        """
        Return a typed version of ndarray `p`.
        """
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

    def visibility_illumination_matrix(self, p):
        r"""Produce the "kernel" of illumination that is integrated against the pixel map.

        Args:
            p (ndarray):

        The kernel is composed of a product of cosines: the illumination is
        proportional to :math:`\vec{n} \cdot \vec{n_s}`, where :math:`\vec{n}`
        is the pixel normal and :math:`\vec{n_s}` is the vector to the star.
        The visibility is proportional to :math:`\vec{n} \cdot \vec{n_o}`,
        where :math:`\vec{n_o}` is the vector to the observer.  The
        contribution of any pixel of value `p` to the lightcurve is therefore
        :math:`p (\vec{n} \cdot \vec{n_s}) (\vec{n} \cdot \vec{n_o})` if both
        :math:`\vec{n} \cdot \vec{n_s}` and :math:`\vec{n} \cdot \vec{n_o}` are
        > 0, and zero otherwise.  So, we need to evaluate these dot products.

        Fix a coordinate system in which to evaluate these dot products
        as follows:

        The orbit is in the x-y plane (so :math:`\hat{L} \parallel \hat{z}`),
        with the x-axis pointing along superior conjunction.  The observer is
        therefore in the x-z plane, and has an inclination angle :math:`\iota`
        in :math:`[0, \pi/2]` to the orbital plane.  So :math:`\vec{n_o} =
        (-\sin(\iota), 0, \cos(\iota))`.

        The planet star vector, :math:`\vec{n_s}`, is given by :math:`\vec{n_s} =
        (-\sin(x_i), -\cos(x_i), 0)`, where :math:`x_i` is the orbital phase.  If
        the orbit has phase :math:`x_{i0}` at `t = 0`, then
        :math:`n_s = R_z(2\pi/P_\mathrm{orb}t + x_{i0}) \cdot (-1,0,0)`.

        For the normal vector to the planet, we must describe the series of
        rotations that maps the orbital coordinate system into the planet-centred
        coordinate system.  Imagine that the planet spin axis is at first aligned
        with the z-axis.  We apply :math:`R_y(\mathrm{obl})`, where `obl` is the
        obliquity angle in :math:`[0, \pi/2]`, and then
        :math:`R_z(\phi_\mathrm{rot})`, where :math:`\phi_\mathrm{rot}` is the
        azimuthal angle of the planet's spin axis in :math:`[0, 2\pi]`.  Now the
        planet's spin axis points to :math:`S =
        (\cos(\phi_\mathrm{rot})*\sin(\mathrm{obl}),
        \sin(\phi_\mathrm{rot})*\sin(\mathrm{obl}), \cos(\mathrm{obl}))`.  At time
        :math:`t`, the planet's normals are given by :math:`n(t) =
        R_S(2\pi/P_\mathrm{rot} t) n(0)`, where :math:`n(0) =
        R_z(\phi_\mathrm{rot}) R_y(\mathrm{obl}) n`, with `n` the body-centred
        normals to the pixels.  We can now evaluate dot products in the fixed
        orbital frame.

        In principle, we are done, but there is an efficiency consideration.  For
        each time at which we have an observation, we need to perform rotations on
        the various vectors in the problem.  There are, in general, a large number
        of n vectors (we have many pixels covering the planet's surface), but only
        one n_o and n_s vector.  It will be more efficient to apply rotations only
        to the n_o and n_s vectors instead of rotating the many, many, n vectors.
        (This is equivalent to performing the dot products in the planet-centred
        frame; we could have done that from the beginning, but it is easier to
        describe the geometry in the orbit-centred frame.)  We can accomplish this
        via a trick: when vectors in a dot product are rotated, the rotations can
        be moved from one vector to the other:

        .. math::

            (R_A a) (R_B b) = < R_A a | R_B b > = < R_B^{-1} R_A a | b > = < a | R_A^{-1} R_B b >

        So, instead of rotating the normal vectors to the planet pixels by

        .. math::

            n(t) = R_S(\omega_\mathrm{rot}(t)) R_z(\phi_\mathrm{rot}) R_y(\mathrm{obl}) n

        We can just rotate the vectors that are inner-producted with these vectors by

        .. math::

            R_\mathrm{inv} = R_y(-\mathrm{obl}) R_z(-\phi_\mathrm{rot}) R_S(-\omega_\mathrm{rot})

        """

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

    def analytic_visibility_illumination_matrix(self, p):
        omega_rot = 2.0*np.pi/self.rotation_period(p)
        obl = self.obl(p)
        obl_orientation = self.obl_orientation(p)
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

        obl_orientation = inv_logit(p['logit_obl_orientation'], 0.0, 2.0*np.pi)
        cos_obl_orientation = np.cos(obl_orientation)
        sin_obl_orientation = np.sin(obl_orientation)

        omega_rot = 2.0*np.pi/np.exp(p['log_rotation_period'])

        S = np.array([cos_obl_orientation*sin_obl, sin_obl_orientation*sin_obl, cos_obl])

        spin_rot_quat = quaternion_multiply(rotation_quaternions(np.array([0.0, 1.0, 0.0]), -obl), \
                                            rotation_quaternions(np.array([0.0, 0.0, 1.0]), -obl_orientation))

        rot_quats = rotation_quaternions(S, -omega_rot*times)

        return quaternion_multiply(spin_rot_quat,rot_quats)

    def _observer_normal_orbit_coords(self, p):
        p = self.to_params(p)

        cos_inc = inv_logit(p['logit_cos_inc'], 0.0, 1.0)
        sin_inc = np.sqrt(1.0 - cos_inc*cos_inc)

        return np.array([-sin_inc, 0.0, cos_inc])

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
        lp += flat_logit_log_prior(p['logit_obl_orientation'], low=0, high=2*np.pi)
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
            return np.diag(np.square(self.sigma_reflectance))
        else:
            return np.diag(1.0/np.square(self.sigma_reflectance))

    def hpmap(self, p):
        p = self.to_params(p)

        if self.map_parameterization == 'pix':
            return p['map']
        elif self.map_parameterization == 'alm':
            complex_alms = p['map'][:self.nmap_params//2] + 1j*p['map'][self.nmap_params//2:]
            return hp.alm2map(complex_alms, self.nside, lmax=self.lmax, mmax=self.mmax, verbose=False)
        else:
            raise RuntimeError("Unrecognized map parameterization {}".format(self.map_parameterization))

    def lightcurve(self, p):
        p = self.to_params(p)

        V = self.resolved_visibility_illumination_matrix(p)
        hpmap = self.hpmap(p)

        map_lc = np.dot(V, hpmap)

        return map_lc

    def log_map_prior(self, p):
        p = self.to_params(p)

        hpmap = self.hpmap(p)
        if hpmap.min() < 0. or hpmap.max() > 1.:
            return -np.inf

        log_mapp = map_logprior_cl(hpmap, p['mu'], self.sigma(p), self.wn_rel_amp(p), self.spatial_scale(p))
        return log_mapp

    def loglikelihood(self, p):
        p = self.to_params(p)

        errsc = self.error_scale(p)

        V = self.resolved_visibility_illumination_matrix(p)

        log_mapp = self.log_map_prior(p)

        map_lc = np.dot(V, self.hpmap(p))

        residual = self.reflectance - map_lc
        sigmas = self.sigma_reflectance

        chi2 = np.sum(np.square(residual/(errsc*sigmas)))

        Nd = residual.shape[0]

        log_datap = -0.5*Nd*np.log(2.0*np.pi) - np.sum(np.log(errsc*sigmas)) - 0.5*chi2

        return log_datap + log_mapp

    def __call__(self, p):
        logp = self.loglikelihood(p)
        if np.isfinite(logp):
            logp += self.log_prior(p)
        return logp

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


# Define separate prior and likelihood classes; useful for parallel-tempered MCMC
class IlluminationMapPrior(IlluminationMapPosterior):
    def __init__(self, *args, **kwargs):
        super(IlluminationMapPrior, self).__init__(*args, **kwargs)

    def __call__(self, p):
        if np.isfinite(self.loglikelihood(p)):
            return self.log_prior(p)
        else:
            return -np.inf

class IlluminationMapLikelihood(IlluminationMapPosterior):
    def __init__(self, *args, **kwargs):
        super(IlluminationMapLikelihood, self).__init__(*args, **kwargs)

    def __call__(self, p):
        return self.loglikelihood(p)
