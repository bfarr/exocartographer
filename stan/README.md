# An Implementation of Exocartographer in Stan

This directory contains a [Stan](http://mc-stan.org) file that implements an
exocartographer-like algorithm.  Because Stan does not natively understand
[healpix](https://healpix.jpl.nasa.gov/), it is necessary to do some
pre-computation in order to use the Stan sampler.  For a full example, see
[chemex](https://github.com/benjaminpope/chemex); here are some brief notes in
the use of this code.

Stan takes a ``data`` dictionary for its sampling.  That dictionary should
contain entries with the the following keys:

* ``lmax``: the maximum angular wavenumber in the analysis.  This is only used
 to set a prior on the GP regularization length/angle scale.

* ``nalm``: the number of coefficients in the spherical harmonic expansion.

* ``npix``: the number of pixels in the maps that will be used to construct
   lightcurves.

* ``ntrend``: the number of "trend" basis vectors that will be fit to the data
   in addition to the surface map.

* ``nobs``: the number of observations in the lightcurve.

* ``l_alm``: an array giving the l number associated to each coefficient
   a_{lm}.  For a particular ordering it could be, for example, ``[0, 1, 1, 1,
   2, 2, 2, 2, 2, ...]``.

* ``sht_matrix``: a matrix of size ``(npix, nalm)`` that represents the
 spherical harmonic transform that takes a_lm coefficients and transforms them
 to pixels.  Note that it must be *real* since Stan does not like complex
 numbers, so you will need to un-pack the transforms from healpix.  A piece of
 Python code that can generate such a matrix for a particular ordering of the
 coefficients (and also generates the ``l_alm`` array tracking which ``l``
 corresponds to which coefficient)
```python
from pylab import *
import healpy
from healpy.sphtfunc import Alm
Ylm_matrix = []
ls = []
for l in range(lmax+1):
    for m in range(0, l+1):
        alm_array = zeros(Alm.getsize(lmax), dtype=np.complex)
        alm_array[Alm.getidx(lmax, l, m)] = 1.0
        Ylm_matrix.append(healpy.alm2map(alm_array, nside))
        ls.append(l)
        if m > 0:
            alm_array[Alm.getidx(lmax, l, m)] = 1.0j
            Ylm_matrix.append(healpy.alm2map(alm_array, nside))
            ls.append(l)
Ylm_matrix = array(Ylm_matrix).T
ls = array(ls, dtype=np.int)
```

* ``pix_area``: the area of each pixel (the maps are in units of flux density,
 so carry a per area).

* ``pix_nhat``: a ``(npix, 3)`` array giving a unit normal vector for each
 pixel.  (See ``healpy.pix2vec``.)

* ``time``: a ``(nobs,)`` array giving the times of the observation.  The origin
  of time defines the meridian of the map

* ``flux``: a ``(nobs,)`` array of fluxes that make up the lightcurve.

* ``sigma_flux``: a ``(nobs,)`` array of uncertainties on the fluxes.  The code
 fits for an errorbar scale factor, called ``nu``, so ``sigma_flux_true =
 nu*sigma_flux``.

* ``trend_basis``: a ``(nobs, ntrend)`` matrix giving the trending basis vectors
 that will be simultaneously fit in the lightcurve with the maps.

* ``Pmin``: the minimum boundary for the rotational period.

* ``Pmax``: the maximum boundary for the rotational period.

* ``mu_P``: the peak of the Gaussian prior on period.

* ``sigma_P``: the s.d. of the Gaussian prior on period.

TODO: More information about Stan sampler.
