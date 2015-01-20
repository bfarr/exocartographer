import numpy as np
from scipy.special import sph_harm


def geometry(t, initial_star_lon, initial_obs_lon, omega_orb, omega_rot):
    # edge-on, zero-obliquity planet
    star_lat = 0.5 * np.pi * np.ones_like(t)
    obs_lat = 0.5 * np.pi * np.ones_like(t)
    star_lon = initial_obs_lon - (omega_rot - omega_orb) * t
    obs_lon = initial_obs_lon - omega_rot * t
    return star_lat, star_lon, obs_lat, obs_lon


def illumination(lat, lon, star_lat, star_lon):
    I = np.cos(lat)[..., np.newaxis] * np.cos(star_lat) +\
        np.sin(star_lat * np.cos(lon[..., np.newaxis] - star_lon)) *\
        np.sin(lat)[..., np.newaxis]
    I[I < 0] = 0

    return I


def visability(lat, lon, obs_lat, obs_lon):
    V = np.sin(lat)[..., np.newaxis] *\
        np.sin(obs_lat) * np.cos(lon[..., np.newaxis] - obs_lon) +\
        np.cos(lat)[..., np.newaxis] * np.cos(obs_lat)
    V[V < 0] = 0

    return V


def diffuse_kernel(lat, lon, star_lat, star_lon, obs_lat, obs_lon):
    I = illumination(lat, lon, star_lat, star_lon)
    V = visability(lat, lon, obs_lat, obs_lon)

    return I * V


def albedo_map(lat, lon, max_l, coefficients):
    albedo = np.zeros_like(lat)

    i = 0
    for l in range(max_l+1):
        for m in range(-l, l+1):
            albedo += coefficients[i] * np.real(sph_harm(m, l, lon, lat))
            i += 1

    return albedo


def lightcurve(t, lat, lon, max_l, coefficients,
               initial_star_lon, initial_obs_lon, omega_orb, omega_rot):
    """
    Diffuse reflection from a zero-obliquity, edge-on planet.
    """
    albedo = albedo_map(lat, lon, max_l, coefficients)

    star_lat, star_lon, obs_lat, obs_lon = geometry(t, initial_star_lon,
                                                    initial_obs_lon, omega_orb,
                                                    omega_rot)

    K = diffuse_kernel(lat, lon, star_lat, star_lon, obs_lat, obs_lon)

    flux = np.sum(albedo[..., np.newaxis] * K * np.sin(lat)[..., np.newaxis],
                  axis=(0, 1))

    return flux
