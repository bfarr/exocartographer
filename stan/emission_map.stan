functions {
  real map_flux(vector pix_map, vector[] pix_nhat, real t, real P, real cos_iota) {
    real sin_iota = sqrt(1.0-cos_iota*cos_iota);
    real omega = 2*pi()*t/P;
    vector[3] nhat = to_vector({cos(omega)*sin_iota, sin(omega)*sin_iota, cos_iota});

    int npix = num_elements(pix_map);

    vector[npix] vis;

    for (i in 1:npix) {
      vis[i] = nhat'*pix_nhat[i];
      if (vis[i] < 0) vis[i] = 0.0;
    }

    return vis'*pix_map;
  }
}

data {
  int lmax;
  int nalm;
  int npix;
  int ntrend;

  int nobs;

  int l_alm[nalm]; /* Gives the l-values of each component in alm space. */

  matrix[npix, nalm] sht_matrix;

  vector[3] pix_nhat[npix];

  real time[nobs];
  real flux[nobs];
  real sigma_flux[nobs];

  matrix[nobs, ntrend] trend_basis;

  /* Impose some boundaries on the period because otherwise the search problem
  /* is too hard. */
  real Pmin;
  real Pmax;

  real cos_iota_min;
  real cos_iota_max;

  vector[nalm] map_mu;
  vector[nalm] map_sigma;

  vector[ntrend] trend_mu;
  vector[ntrend] trend_sigma;
}

parameters {
  real<lower=0> sigma; /* Overall scatter in log-flux. */
  real<lower=0.1, upper=lmax> lambda; /* Scale (in l-space) of correlations. */

  real<lower=cos_iota_min,upper=cos_iota_max> cos_iota; /* cosine(inclination) */
  real<lower=Pmin, upper=Pmax> P; /* Rotation period. */

  vector[nalm] dmap; /* flux from each Ylm (can be negative because added to trend to generate observed flux.) */

  vector[ntrend] dtrend;
}

transformed parameters {
  vector[nalm] map = map_mu + map_sigma .* dmap;
  vector[ntrend] c_trend = trend_mu + trend_sigma .* dtrend;
}

model {
  /* Flat prior on mu */

  /* Log-normal prior on sigma. */
  sigma ~ lognormal(0,1);

  /* Log-normal prior on lambda. */
  lambda ~ lognormal(0,1);

  /* Flat prior on cos-theta. */
  /* Flat prior on period. */

  /* N(0,1) on c_trend */
  c_trend ~ normal(0,1);

  {
    /* Gaussian process prior on map. */
    vector[nalm] sqrt_Cl;

    for (i in 1:nalm) {
      sqrt_Cl[i] = sigma*exp(-l_alm[i]*l_alm[i]/(4.0*lambda*lambda));
    }

    map ~ normal(0.0, sqrt_Cl); /* Zero-mean because mean level is taken care of by trend. */
  }

  /* Likelihood */
  {
    vector[npix] pix_map = sht_matrix*map;
    vector[nobs] trend = trend_basis*c_trend;
    vector[nobs] mflux;

    for (i in 1:nobs) {
      mflux[i] = map_flux(pix_map, pix_nhat, time[i], P, cos_iota);
    }

    flux ~ normal(mflux + trend, sigma_flux);
  }
}

generated quantities {
  vector[npix] pix_map = sht_matrix*map;
  vector[nobs] trend = trend_basis*c_trend;
  vector[nobs] lightcurve;

  {
    vector[nobs] mflux;

    for (i in 1:nobs) {
      mflux[i] = map_flux(pix_map, pix_nhat, time[i], P, cos_iota);
    }

    lightcurve = mflux + trend;
  }
}
