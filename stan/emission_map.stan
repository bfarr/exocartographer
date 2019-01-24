functions {
  vector visibility(vector[] pix_nhat, real pix_area, real t, real P, real cos_iota) {
    real sin_iota = sqrt(1.0-cos_iota*cos_iota);
    real omega = -2*pi()*t/P;
    vector[3] nhat = to_vector({cos(omega)*sin_iota, sin(omega)*sin_iota, cos_iota});

    int npix = size(pix_nhat);

    vector[npix] vis;

    for (i in 1:npix) {
      vis[i] = pix_area*nhat'*pix_nhat[i];
      if (vis[i] < 0) vis[i] = 0.0;
    }

    return vis;
  }

  matrix design_matrix(matrix sht_matrix, matrix trend_basis, vector[] pix_nhat, real pix_area, real[] times, real P, real cos_iota) {
    int npix = dims(sht_matrix)[1];
    int nalm = dims(sht_matrix)[2];
    int nobs = dims(trend_basis)[1];
    int ntrend = dims(trend_basis)[2];

    matrix [nobs+nalm+ntrend, nalm+ntrend] M = rep_matrix(0.0, nobs+nalm+ntrend, nalm+ntrend);

    for (i in 1:nobs) {
      vector[npix] vis = visibility(pix_nhat, pix_area, times[i], P, cos_iota);
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

  vector measurement_invsigma(vector flux_error, vector sqrt_C_l, vector sigma_trend) {
    int nobs = num_elements(flux_error);
    int nalm = num_elements(sqrt_C_l);
    int ntrend = num_elements(sigma_trend);

    vector[nobs+nalm+ntrend] mprec;

    mprec[1:nobs] = 1.0 ./ flux_error;
    mprec[nobs+1:nobs+nalm] = 1.0 ./ sqrt_C_l;
    mprec[nobs+nalm+1:] = 1.0 ./ sigma_trend;

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
}

data {
  int lmax;
  int nalm;
  int npix;
  int ntrend;

  int nobs;

  int l_alm[nalm]; /* Gives the l-values of each component in alm space. */

  matrix[npix, nalm] sht_matrix;

  real pix_area;
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
}

parameters {
  real<lower=0> sigma; /* Overall scatter in log-flux. */
  real<lower=0.5, upper=2*lmax> lambda; /* Scale (in l-space) of correlations. */

  real<lower=0,upper=1> cos_iota; /* cosine(inclination) */
  real<lower=Pmin, upper=Pmax> P; /* Rotation period. */

  real<lower=0.1, upper=10.0> nu; /* Errorbar scaling. */
}

transformed parameters {
  /* Gaussian process prior on map */
  vector[nalm] sqrt_Cl;

  for (i in 1:nalm) {
    /* The 0.01/(2*l_alm[i]+1) gives a 1% white noise amplitude in each l mode,
    /* and should make the GP much more stable. */
    sqrt_Cl[i] = sigma*(exp(-l_alm[i]*l_alm[i]/(4.0*lambda*lambda)) + 0.01/(2*l_alm[i]+1));
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

  /* Errorbar scaling. */
  nu ~ lognormal(0.0, 1.0);

  /* Likelihood */
  {
    matrix[nobs+nalm+ntrend, nalm+ntrend] M = design_matrix(sht_matrix, trend_basis, pix_nhat, pix_area, time, P, cos_iota);
    vector[nobs+nalm+ntrend] msigma = measurement_invsigma(nu*to_vector(sigma_flux), sqrt_Cl, sigma_trend);
    vector[nobs+nalm+ntrend] meas = measurements(to_vector(flux), rep_vector(0.0, nalm), rep_vector(0.0, ntrend));
    vector[nobs+nalm+ntrend] scaled_meas = meas .* msigma;
    matrix[nobs+nalm+ntrend, nalm+ntrend] scaled_M = diag_pre_multiply(msigma, M);
    matrix[nobs+nalm+ntrend, nalm+ntrend] sm_Q = qr_thin_Q(scaled_M);
    matrix[nalm+ntrend, nalm+ntrend] sm_R = qr_thin_R(scaled_M);
    vector[nalm+ntrend] best_fit = mdivide_right_tri_low(scaled_meas' * sm_Q, sm_R')';

    vector[nobs+nalm+ntrend] resid = meas - M*best_fit;

    target += -0.5*sum(resid .* msigma .* msigma .* resid);
    target += -sum(log(nu*to_vector(sigma_flux))) - sum(log(sqrt_Cl)) - sum(log(sigma_trend)) - sum(log(diagonal(sm_R)));
  }
}

generated quantities {
  vector[nalm] alm_map;
  vector[npix] pix_map;
  vector[nobs] trend;
  vector[nobs] lightcurve;

  {
    vector[ntrend] sigma_trend = rep_vector(1.0, ntrend);
    matrix[nobs+nalm+ntrend, nalm+ntrend] M = design_matrix(sht_matrix, trend_basis, pix_nhat, pix_area, time, P, cos_iota);
    vector[nobs+nalm+ntrend] msigma = measurement_invsigma(nu*to_vector(sigma_flux), sqrt_Cl, sigma_trend);
    vector[nobs+nalm+ntrend] meas = measurements(to_vector(flux), rep_vector(0.0, nalm), rep_vector(0.0, ntrend));
    vector[nobs+nalm+ntrend] scaled_meas = meas .* msigma;
    matrix[nobs+nalm+ntrend, nalm+ntrend] scaled_M = diag_pre_multiply(msigma, M);
    matrix[nobs+nalm+ntrend, nalm+ntrend] sm_Q = qr_thin_Q(scaled_M);
    matrix[nalm+ntrend, nalm+ntrend] sm_R = qr_thin_R(scaled_M);
    vector[nalm+ntrend] best_fit = mdivide_right_tri_low(scaled_meas' * sm_Q, sm_R')';

    vector[nalm+ntrend] udraw;
    vector[nalm+ntrend] draw;
    vector[ntrend] trend_draw;

    for (i in 1:nalm+ntrend) {
      udraw[i] = normal_rng(0, 1);
    }

    draw = best_fit + mdivide_right_tri_low(udraw', sm_R')';
    alm_map = draw[1:nalm];
    trend_draw = draw[nalm+1:];
    pix_map = sht_matrix*alm_map;
    trend = trend_basis*trend_draw;
    lightcurve = (M*draw)[1:nobs];
  }
}
