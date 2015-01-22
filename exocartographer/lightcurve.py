import numpy as np

class Likelihood(object):
    """
    A likelihood class object for exoplanet mapping from lightcurves.
    """
    
    def __init__(self, data, template, lat, lon, dt, max_l=1, init_obs_lon=0.):
        """
        Initialize the posterior object.
        
        :param data: A 1-D array containing the flux timeseries to analyze.
        
        :param template: A function for generating model lightcurves, given an array of parameters.
        
        :param lat: Array of pixel latitudes.
        
        :param lon: Array of pixel longitudes.
        """
        
        self.dt = dt
        self.data = data
        self.template = template
    
        self.max_l = max_l
        self.N = len(self.data)
        self.init_obs_lon = init_obs_lon
        self.lat = lat
        self.lon = lon
        self.times = np.linspace(0, self.N*dt, self.N, endpoint=False)
        
    @property
    def dtype(self):
        """
        Give the numpy datatype of the parameters.
        """
        return np.dtype([('a1', np.float),
                         ('a2', np.float),
                         ('a3', np.float),
                         ('a4', np.float),
                         ('omega_orb', np.float),
                         ('omega_rot', np.float),
                         ('init_star_lon', np.float),
                         ('sigma_n', np.float)])

    @property
    def pnames(self):
        """
        Give LaTeX names for the parameters.
        """
        return [r'$a_1$',
                r'$a_2$',
                r'$a_3$',
                r'$a_4$',
                r'$\omega_\mathrm{orb}$',
                r'$\omega_\mathrm{rot}$',
                r'$\theta_\mathrm{star}$',
                r'$\sigma_\mathrm{n}$']

    def to_params(self, p):
        """
        Convert the array ``p`` to a named-array representing parameters.
        """
        return p.view(self.dtype).squeeze()
    
    def likelihood(self, p=None):
        """
        Calculate the likelihood at ``p``.
        """
        p = self.to_params(p)

        coefficients = [p['a1'], p['a2'], p['a3'], p['a4']]
        sigma_n = p['sigma_n']

        residual = self.data - self.template(self.times, self.lat, self.lon, self.max_l,
                                             coefficients, p['init_star_lon'], self.init_obs_lon,
                                             p['omega_orb'], p['omega_rot'])

        return np.sum(-residual*residual/(2*sigma_n*sigma_n)) - self.N*np.log(np.sqrt(2*np.pi)*sigma_n)
    
    def __call__(self, p):
        return self.likelihood(p)
