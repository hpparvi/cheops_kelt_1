from numba import njit, prange
from numpy import atleast_1d, asarray, zeros, pi, cos, sin, fabs


@njit(parallel=True)
def ellipsoidal_variation(a, t0, p, x_is_time, x):
    a = atleast_1d(asarray(a))
    t0 = atleast_1d(asarray(t0))
    p = atleast_1d(asarray(p))
    npv = a.size
    npt = x.size
    flux = zeros((npv, npt))
    for i in prange(npv):
        if x_is_time:
            for j in range(npt):
                f = 2.0 * pi * (x[j] - t0[i]) / p[i]
                flux[i, j] = -a[i] * cos(2.0 * f)
        else:
            for j in range(npt):
                flux[i, j] = -a[i] * cos(2.0 * x[j])
    return flux


@njit(parallel=True)
def doppler_boosting(a, t0, p, x_is_time, x):
    a = atleast_1d(asarray(a))
    t0 = atleast_1d(asarray(t0))
    p = atleast_1d(asarray(p))
    npv = a.size
    npt = x.size
    flux = zeros((npv, npt))
    for i in range(npv):
        if x_is_time:
            for j in range(npt):
                f = 2.0 * pi * (x[j] - t0[i]) / p[i]
                flux[i, j] = a[i] * sin(f)
        else:
            for j in range(npt):
                flux[i, j] = a[i] * sin(x[j])
    return flux


@njit(parallel=True)
def ev_and_db(aev, adb, t0, p, x_is_time, x):
    aev = atleast_1d(asarray(aev))
    adb = atleast_1d(asarray(adb))
    t0 = atleast_1d(asarray(t0))
    p = atleast_1d(asarray(p))
    npv = aev.size
    npt = x.size
    flux = zeros((npv, npt))
    for i in prange(npv):
        if x_is_time:
            for j in range(npt):
                f = 2.0 * pi * (x[j] - t0[i]) / p[i]
                flux[i, j] = -aev[i] * cos(2.0 * f) + adb[i] * sin(f)
        else:
            for j in range(npt):
                flux[i, j] = -aev[i] * cos(2.0 * x[j]) + adb[i] * sin(x[j])
    return flux


@njit(parallel=True)
def emission(aratio, fr_night, fr_day, offset, t0, p, x_is_time, x):
    aratio = atleast_1d(asarray(aratio))
    fr_night = atleast_1d(asarray(fr_night))
    fr_day = atleast_1d(asarray(fr_day))
    offset = atleast_1d(asarray(offset))
    t0 = atleast_1d(asarray(t0))
    p = atleast_1d(asarray(p))
    npv = aratio.size
    npt = x.size
    flux = zeros((npv, npt))
    for i in prange(npv):
        if x_is_time:
            for j in range(npt):
                f = 2.0 * pi * (x[j] - t0[i]) / p[i]
                flux[i, j] = aratio[i] * (fr_night[i] + (fr_day[i] - fr_night[i]) * 0.5 * (1.0 - cos(f + offset[i])))
        else:
            for j in range(npt):
                flux[i, j] = aratio[i] * (fr_night[i] + (fr_day[i] - fr_night[i]) * 0.5 * (1.0 - cos(x[j] + offset[i])))
    return flux


@njit(parallel=True)
def lambert_phase_function(a, ar, ga, t0, p, x_is_time, x):
    a = atleast_1d(asarray(a))  # Scaled semi-major axis
    ar = atleast_1d(asarray(ar))  # Area ratio
    ga = atleast_1d(asarray(ga))  # geometric albedo
    t0 = atleast_1d(asarray(t0))
    p = atleast_1d(asarray(p))
    npv = a.size
    npt = x.size
    flux = zeros((npv, npt))
    for i in prange(npv):
        amp = ar[i] * ga[i] / a[i]**2
        if x_is_time:
            for j in range(npt):
                f = 2.0 * pi * (x[j] - t0[i]) / p[i]
                alpha = fabs(f % (2.0 * pi) - pi)
                flux[i, j] = amp * (sin(alpha) + (pi - alpha) * cos(alpha)) / pi
        else:
            for j in range(npt):
                alpha = fabs(x[j] % (2.0 * pi) - pi)
                flux[i, j] = amp * (sin(alpha) + (pi - alpha) * cos(alpha)) / pi
    return flux