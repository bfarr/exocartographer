import gp_map as gm
import healpy as hp
import numpy as np
import scipy.stats as ss
from util import logit, inv_logit, flat_logit_log_prior

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

    _param_names = ['mu', 'log_sigma', 'logit_wn_rel_amp', 'logit_spatial_scale',
                    'log_rotation_period', 'log_orbital_period',
                    'logit_phi_orb', 'logit_cos_obl', 'logit_phi_rot',
                    'logit_cos_inc']
    
    @property
    def dtype(self):
        return np.dtype([(n, np.float) for n in self._param_names])
    
    @property
    def dtype_map(self):
        typel = [(n, np.float) for n in self._param_names]
        typel.append(('albedo_map', np.float, self.npix))
        return np.dtype(typel)

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
        if isinstance(p, np.ndarray):
            if p.dtype == self.dtype or p.dtype == self.dtype_map:
                return p
            else:
                p = p.view(float)
                if p.shape[-1] == self.nparams:
                    return p.view(self.dtype).squeeze()
                else:
                    return p.view(self.dtype_map).squeeze()
        else:
            p = np.atleast_1d(p)
            return self.to_params(p)

    def params_map_to_params(self, pm):
        pm = self.to_params(pm)
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

        return area*cos_factors

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
        p = self.to_params(p)
        return p['albedo_map']*self.resolved_visibility_illumination_matrix(p)

    def lightcurve_map(self, p):
        return np.sum(self.visibility_illumination_maps(p), axis=1)

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

    def log_pdata_map(self, p):
        p = self.to_params(p)

        lightcurve = self.lightcurve_map(p)

        return np.sum(ss.norm.logpdf(self.intensity, loc=lightcurve, scale=self.sigma_intensity))

    def log_pmap_map(self, p):
        p = self.to_params(p)

        sigma = np.exp(p['log_sigma'])
        wn_rel_amp = self.wn_rel_amp(p)
        lambda_spatial = self.spatial_scale(p)

        return gm.map_logprior(p['albedo_map'], p['mu'], sigma, wn_rel_amp, lambda_spatial)

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

        dover_sigma = self.intensity/self.sigma_intensity/self.sigma_intensity

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

        residual = self.intensity - map_lc
        sigmas = self.sigma_intensity

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
        return self.log_prior(p) + self.log_pdata_map(p) + self.log_pmap_map(p)

    def draw_map(self, p):
        V = self.resolved_visibility_illumination_matrix(p)
        gamma = self.gamma_matrix(p, V)
        mbar = self.mbar(p, gamma, V)

        return np.random.multivariate_normal(mbar, gamma)

    def sub_observer_latlong(self, p, times):
        no = self._observer_normal_orbit_coords(p)

        body_quats = self._body_frame_quaternions(p, times)

        no_body = rotate_vector(body_quats, no)

        colats = np.arccos(no_body[:,2])
        longs = np.arctan2(no_body[:,1], no_body[:,0])

        longs[longs < 0] += 2*np.pi

        return np.pi/2.0-colats, longs
