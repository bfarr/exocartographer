import numpy as np

# VIEWING GEOMETRY FUNCTION: "viewgeom"
# Coded by Joel Schwartz (Updated from script by Clara Sekowski)
# 6/1/2016 Version
# minor tweaks by Nick Cowan
#
# Code for calculating time-dependent trig expressions for planetary
# sub-observer and sub-stellar locations.
#
# Ordered Input: array of discrete time values, rotational and
# orbital angular frequencies, obliquity, inclination, and orbital
# solstice phase (xisol).
#
# (NOTE: "xisol" is defined as the orbital angle from superior conjunction to
# the "summer solstice" for the northern hemisphere.)
#
# Output: array of numerical trig functions, one time per row,
# with columns organized as follows:
#
# [sin ThObs, cos ThObs, sin PhiObs, cos PhiObs, sin ThSt, cos ThSt, sin PhiSt, cos PhiSt]
def viewgeom(times, wrot, worb, obq, inc, xisol, xi0):
    steps = times.shape[0]      # Number of time steps from input array
    xi = worb*times + xi0
    phiGen = wrot*times         # General expression for PhiObs (without negative sign)

    cThObs = (np.cos(inc)*np.cos(obq)) + (np.sin(inc)*np.sin(obq)*np.cos(xisol))
    cThObsfull = np.zeros(steps) + cThObs       # Duplicating cos ThObs (constant)
    sThObs = (1.0 - (cThObs**2.0))**0.5

    sThObsfull = np.zeros(steps) + sThObs       # Duplicating sin ThObs (constant)
    cThSt = np.sin(obq)*np.cos(xi - xisol)
    sThSt = (1.0 - (cThSt**2.0))**0.5

    sol_r = (xisol % (2.0*np.pi))                # Solstice modulo 360 degrees
    inc_rd = round(inc,8)                   # Rounded inclination (for better comparison)
    p_obq_rd = round((np.pi - obq),8)          # Rounded 180 degrees - obliquity (for better comparison)

    cond_face = ((inc == 0) and ((obq == 0) or (obq == np.pi)))                    # "Pole-observer" I: face-on inclination
    cond_north = ((sol_r == 0) and ((inc == obq) or (inc_rd == -p_obq_rd)))     # Ditto II: North pole view
    cond_south = ((xisol == np.pi) and ((inc_rd == p_obq_rd) or (inc == -obq)))    # Ditto III: South pole view

    if cond_face or cond_north or cond_south:
        if (obq == (np.pi/2.0)):
            aII = np.sin(xi)*np.cos(xisol)      # Special "double-over-pole" time-dependent factor "aII"
            cPhiSt = np.ones(steps)
            sPhiSt = np.zeros(steps)
            g_i = (sThSt != 0)             # Excluding "star-over-pole" situations (g_i = "good indicies")
            cPhiSt[g_i] = (-np.sin(phiGen[g_i])*aII[g_i])/sThSt[g_i]
            sPhiSt[g_i] = (-np.cos(phiGen[g_i])*aII[g_i])/sThSt[g_i]
        else:
            aI = np.cos(xi)*np.cos(obq)       # Alternate "observer-over-pole" time-dependent factor "aI"
            bI = np.sin(xi)                # Ditto "bI"
            cPhiSt = ((np.cos(phiGen)*aI) + (np.sin(phiGen)*bI))/sThSt
            sPhiSt = ((-np.sin(phiGen)*aI) + (np.cos(phiGen)*bI))/sThSt
    else:
        a = (np.sin(inc)*np.cos(xi)) - (cThObs*cThSt)         # Time-dependent factor "a"
        b = ((np.sin(inc)*np.sin(xi)*np.cos(obq) - np.cos(inc)*np.sin(obq)*np.sin(xi - xisol)))  # Ditto "b"; 4_28_15 includes missing negative sign
        if (obq == (np.pi/2.0)):
            cPhiSt = np.ones(steps)
            sPhiSt = np.zeros(steps)
            g_i = (sThSt != 0)             # Excluding "star-over-pole" situations (g_i = "good_indicies")
            cPhiSt[g_i] = ((np.cos(phiGen[g_i])*a[g_i]) + (np.sin(phiGen[g_i])*b[g_i]))/(sThObs*sThSt[g_i])
            sPhiSt[g_i] = ((-np.sin(phiGen[g_i])*a[g_i]) + (np.cos(phiGen[g_i])*b[g_i]))/(sThObs*sThSt[g_i])
        else:
            cPhiSt = ((np.cos(phiGen)*a) + (np.sin(phiGen)*b))/(sThObs*sThSt)
            sPhiSt = ((-np.sin(phiGen)*a) + (np.cos(phiGen)*b))/(sThObs*sThSt)

    trigvals = np.column_stack((sThObsfull, cThObsfull, np.sin(-phiGen), np.cos(-phiGen),
                                sThSt, cThSt, sPhiSt, cPhiSt))  # Compiling output array
    return trigvals

# EXTRA KERNEL CODE
# Joel Schwartz
# 6/1/2016
# Minor tweak by Nick Cowan
# 6/1/2016
# Full Kernel (no marginalization over theta or phi)
def kernel(Sth, Cth, Sph, Cph, trigA):  # Arrays: Sin theta, Cos theta, Sin phi, Cos phi, viewgeom values (single phase)
    Vis = (Sth[..., np.newaxis]*trigA[:, 0]*(Cph[..., np.newaxis]*trigA[:, 3] + Sph[..., np.newaxis]*trigA[:, 2])) + (Cth[..., np.newaxis]*trigA[:, 1])
    Vis[Vis < 0] = 0

    Ilu = (Sth[..., np.newaxis]*trigA[:, 4]*(Cph[..., np.newaxis]*trigA[:, 7] + Sph[..., np.newaxis]*trigA[:, 6])) + (Cth[..., np.newaxis]*trigA[:, 5])
    Ilu[Ilu < 0] = 0

    return (Vis * Ilu) / np.pi
