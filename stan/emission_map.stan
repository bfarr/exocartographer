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

  vector visibility(vector[] pix_nhat, real t, real P, real cos_iota) {
    real sin_iota = sqrt(1.0-cos_iota*cos_iota);
    real omega = 2*pi()*t/P;
    vector[3] nhat = to_vector({cos(omega)*sin_iota, sin(omega)*sin_iota, cos_iota});

    int npix = size(pix_nhat);

    vector[npix] vis;

    for (i in 1:npix) {
      vis[i] = nhat'*pix_nhat[i];
      if (vis[i] < 0) vis[i] = 0.0;
    }

    return vis;
  }

  matrix design_matrix(matrix sht_matrix, matrix trend_basis, vector[] pix_nhat, real[] times, real P, real cos_iota) {
    int npix = dims(sht_matrix)[1];
    int nalm = dims(sht_matrix)[2];
    int nobs = dims(trend_basis)[1];
    int ntrend = dims(trend_basis)[2];

    matrix [nobs+nalm+ntrend, nalm+ntrend] M = rep_matrix(0.0, nobs+nalm+ntrend, nalm+ntrend);

    for (i in 1:nobs) {
      vector[npix] vis = visibility(pix_nhat, times[i], P, cos_iota);
      M[i, 1:nalm] = vis'*sht_matrix;
      M[i, nalm+1:] = trend_basis[i,:];
    }

    /* Now put in the fictituous measurements for the alm prior. */
    for (i in 1:nalm) {
      M[nobs+i, i] = 1.0;
    }

    /* And the fictituous measurements for the trend prior. */
    for (i in 1:ntrend) {
      M[nobs+nalm+i, nalm+i] = 1.0;
    }

    return M;
  }

  vector measurement_precision(vector flux_error, vector sqrt_C_l, vector sigma_trend) {
    int nobs = num_elements(flux_error);
    int nalm = num_elements(sqrt_C_l);
    int ntrend = num_elements(sigma_trend);

    vector[nobs+nalm+ntrend] mprec;

    mprec[1:nobs] = 1.0 ./ (flux_error .* flux_error);
    mprec[nobs+1:nobs+nalm] = 1.0 ./ (sqrt_C_l .* sqrt_C_l);
    mprec[nobs+nalm+1:] = 1.0 ./ (sigma_trend .* sigma_trend);

    return mprec;
  }

  vector measurements(vector flux, vector alm_mean, vector trend_mean) {
    int nobs = num_elements(flux);
    int nalm = num_elements(alm_mean);
    int ntrend = num_elements(trend_mean);

    vector[nobs+nalm+ntrend] meas;

    meas[1:nobs] = flux;
    meas[nobs+1:nobs+nalm] = alm_mean;
    meas[nobs+nalm+1:] = trend_mean;

    return meas;
  }

  matrix precision_matrix(matrix design_matrix, vector measurement_precision) {
    return design_matrix' * diag_pre_multiply(measurement_precision, design_matrix);
  }

  vector best_fit_alm_trend(matrix precision_matrix, matrix design_matrix, vector measurement_precision, vector measurements) {
    return mdivide_left_spd(precision_matrix, design_matrix' * (measurement_precision .* measurements));
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
  real mu_P;
  real sigma_P;

  real cos_iota_min;
  real cos_iota_max;
}

parameters {
  real<lower=0> sigma; /* Overall scatter in log-flux. */
  real<lower=0.1, upper=2*lmax> lambda; /* Scale (in l-space) of correlations. */

  real<lower=cos_iota_min,upper=cos_iota_max> cos_iota; /* cosine(inclination) */
  real<lower=Pmin, upper=Pmax> P; /* Rotation period. */
}

transformed parameters {
  /* Gaussian process prior on map */
  vector[nalm] sqrt_Cl;

  for (i in 1:nalm) {
    sqrt_Cl[i] = sigma*exp(-l_alm[i]*l_alm[i]/(4.0*lambda*lambda));
  }
}

model {
  vector[ntrend] sigma_trend = rep_vector(1.0, ntrend);

  /* Flat prior on mu */

  /* Normal prior on sigma (mild shrinkage). */
  sigma ~ normal(0,1);

  /* Log-normal prior on lambda. */
  {
    /* The choice here is to put l = 1 and l = lmax at the +/- 2-sigma
       boundaries in the log-normal for lambda, since too-small lambdas and
       too-large lambdas lead to divergences */
    real mu_logl;
    real sigma_logl;

    mu_logl = 0.5*(log(lmax) + log(1));
    sigma_logl = 0.25*(log(lmax) - log(1));

    lambda ~ lognormal(mu_logl, sigma_logl);
  }

  /* Flat prior on cos-theta. */
  P ~ normal(mu_P, sigma_P);

  /* Likelihood */
  {
    matrix[nobs+nalm+ntrend, nalm+ntrend] M = design_matrix(sht_matrix, trend_basis, pix_nhat, time, P, cos_iota);
    vector[nobs+nalm+ntrend] mprec = measurement_precision(to_vector(sigma_flux), sqrt_Cl, sigma_trend);
    vector[nobs+nalm+ntrend] meas = measurements(to_vector(flux), rep_vector(0.0, nalm), rep_vector(0.0, ntrend));
    matrix[nalm+ntrend, nalm+ntrend] precmat = precision_matrix(M, mprec);
    vector[nalm+ntrend] best_fit = best_fit_alm_trend(precmat, M, mprec, meas);

    vector[nobs+nalm+ntrend] resid = meas - M*best_fit;

    target += -0.5*sum(resid .* mprec .* resid);
    target += -0.5*log_determinant(precmat);
  }
}

generated quantities {
  vector[npix] pix_map;
  vector[nobs] trend;
  vector[nobs] lightcurve;

  {
    vector[ntrend] sigma_trend = rep_vector(1.0, ntrend);
    matrix[nobs+nalm+ntrend, nalm+ntrend] M = design_matrix(sht_matrix, trend_basis, pix_nhat, time, P, cos_iota);
    vector[nobs+nalm+ntrend] mprec = measurement_precision(to_vector(sigma_flux), sqrt_Cl, sigma_trend);
    vector[nobs+nalm+ntrend] meas = measurements(to_vector(flux), rep_vector(0.0, nalm), rep_vector(0.0, ntrend));
    matrix[nalm+ntrend, nalm+ntrend] precmat = precision_matrix(M, mprec);
    vector[nalm+ntrend] best_fit = best_fit_alm_trend(precmat, M, mprec, meas);

    matrix[nalm+ntrend, nalm+ntrend] Lprecmat = cholesky_decompose(precmat);

    vector[nalm+ntrend] udraw;
    vector[nalm+ntrend] draw;
    vector[nalm] alm_draw;
    vector[ntrend] trend_draw;

    for (i in 1:nalm+ntrend) {
      udraw[i] = normal_rng(0, 1);
    }

    draw = best_fit + mdivide_right_tri_low(udraw', Lprecmat)';
    alm_draw = draw[1:nalm];
    trend_draw = draw[nalm+1:];
    pix_map = sht_matrix*alm_draw;
    trend = trend_basis*trend_draw;
    lightcurve = (M*draw)[1:nobs];
  }
}
