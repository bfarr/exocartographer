import numpy as np
from scipy import special


def geometry(t, initial_star_lon, initial_obs_lon, omega_orb, omega_rot):
    # edge-on, zero-obliquity planet
    star_lat = 0.5 * np.pi * np.ones_like(t)
    obs_lat = 0.5 * np.pi * np.ones_like(t)
    star_lon = initial_obs_lon - (omega_rot - omega_orb) * t
    obs_lon = initial_obs_lon - omega_rot * t
    return star_lat, star_lon, obs_lat, obs_lon


def illumination(lat, lon, star_lat, star_lon):
    lat_size = lat.shape
    n_t = len(star_lon)

    I = np.empty([lat_size[0], lat_size[1], n_t])

    for t_index in range(n_t):
        I[..., t_index] = np.cos(lat) * np.cos(star_lat[t_index]) +\
            np.sin(star_lat[t_index] * np.cos(lon - star_lon[t_index])) *\
            np.sin(lat)

    I[I < 0] = 0
    return I


def visability(lat, lon, obs_lat, obs_lon):
    lat_size = lat.shape
    n_t = len(obs_lon)

    V = np.empty([lat_size[0], lat_size[1], n_t])
    for t_index in range(n_t):
        V[..., t_index] = np.sin(lat) *\
            np.sin(obs_lat[t_index]) * np.cos(lon-obs_lon[t_index]) +\
            np.cos(lat) * np.cos(obs_lat[t_index])
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
            albedo += coefficients[i] *\
                np.real(special.sph_harm(m, l, lon, lat))
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
    star_lat_size = star_lat.shape
    n_t = len(star_lon)

    flux = np.empty(n_t)
    for t_index in range(n_t):
        flux[t_index] = np.sum(albedo * K[..., t_index] * np.sin(lat))

    return flux
