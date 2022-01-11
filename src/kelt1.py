from pathlib import Path

import pandas as pd
import xarray as xa

from astropy.stats import sigma_clip, mad_std
from astropy.table import Table
from matplotlib.pyplot import bar, axhline
from numpy import sin, cos, diff, ones, median, argmin, array, arange, sqrt
from scipy.ndimage import median_filter
from uncertainties import ufloat
import astropy.units as u

from ldtk import LDPSetCreator, SVOFilter

AAOCW, AAPGW = 3.4645669, 7.0866142

def read_mcmc(filename):
    with xa.load_dataset(filename) as ds:
        chain = ds['mcmc_samples'].data.reshape([-1, ds.parameter.size])
        df = pd.DataFrame(chain, columns=ds.parameter)
        df['k'] = sqrt(df.k2)
        return df

filters = (SVOFilter('CHEOPS/CHEOPS.band'), SVOFilter('TESS'),
           SVOFilter('LBT/LUCIFER.H_4302'), SVOFilter('CFHT/Wircam.Ks'),
           SVOFilter('Spitzer/IRAC.I1'), SVOFilter('Spitzer/IRAC.I2'))

filter_names = "CHEOPS TESS H Ks 36um 45um".split()

mj2kg = u.M_jup.to(u.kg)
ms2kg = u.M_sun.to(u.kg)
d2s = u.day.to(u.s)

# KELT-1b parameters
# --------------------
kelt1_m =  ufloat(27.38, 0.93)  # KELT-1b mass in M_Jup

zero_epoch = ufloat(2455914.1628, 0.0023)  # Siverd et al. (2012)
period = ufloat(1.217513, 0.000015)        # Siverd et al. (2012)
aor = ufloat(3.65, 0.03)                   # Beatty  et al. (2020)

b = ufloat(0.195, 0.05)
k2 = ufloat(0.005935, 4.468e-5)

# Stellar parameters
# ------------------
tic = 432549364
star_teff = ufloat(6516,  49)
star_logg = ufloat(4.228,  0.02)
star_z    = ufloat(0.052,  0.079)
star_r    = ufloat(1.471, 0.045)
star_m    = ufloat(1.335, 0.063)
star_rho  = rho = ufloat(0.58, 0.05)

# Eclipse depths and flux ratios
# ------------------------------
edtessb = ufloat(371e-6, 50e-6)  # TESS band from beatty et al.
edtesse = ufloat(304e-6, 75e-6)  # TESS band from von Essen et al.
edh = ufloat(1418e-6, 94e-6)     # H-band from Beatty et al.
edks = ufloat(1600e-6, 190e-6)   # Ks-band from Croll et al.
eds1 = ufloat(1877e-6, 58e-6)    # Spitzer 3.6 um band from Beatty et al. (2019)
eds2 = ufloat(2083e-6, 70e-6)    # Spitzer 4.5 um band from Beatty et al. (2019)

frtessb = ufloat(0.0636, 0.0086)  # TESS band from Beatty et al.
frtesse = ufloat(0.052, 0.013)    # TESS band from von Essen et al.
frh = ufloat(0.243, 0.016)        # H-band from Beatty et al.
frks = ufloat(0.274, 0.033)       # Ks-band from Croll et al.
frs1 = ufloat(0.3218, 0.0099)     # Spitzer 3.6 um band from Beatty et al. (2019)
frs2 = ufloat(0.357, 0.012)       # Spitzer 4.5 um band from Beatty et al. (2019)

# Theoretical ellipsoidal variation amplitudes
# --------------------------------------------
ev_amplitudes = {n:a for n,a in zip(filter_names, [1e-6*ufloat(s) for s in "480+/-30 443+/-28 358+/-23 353+/-22 344+/-22 343+/-22".split()])}

# Doppler beaming amplitudes
# --------------------------
beaming_amplitudes = {n:a for n,a in zip(filter_names, array([6.3e-05, 4.1e-05, 2.5e-05, 1.9e-05, 1.7e-05, 1.6e-05]))}
beaming_uncertainties = {n:a for n,a in zip(filter_names, array([3.2e-06, 2.1e-06, 1.3e-06, 9.2e-07, 8.3e-07, 7.9e-07]))}

# Utility functions
# -----------------

rootdir = Path(__file__).parent.parent
datadir = rootdir / 'data'

def bplot(c):
    qs = c.quantile([0.16, 0.84, 0.025, 0.975, 0.0015, 0.9985, 0.49, 0.51])
    x = arange(qs.shape[1])
    bar(x, qs.values[5]-qs.values[4], bottom=qs.values[4], fc='C0', alpha=0.15)
    bar(x, qs.values[3]-qs.values[2], bottom=qs.values[2], fc='C0', alpha=0.25)
    bar(x, qs.values[1]-qs.values[0], bottom=qs.values[0], fc='C0', alpha=0.45)
    bar(x, 0.00002, bottom=qs.values[6], fc='k')
    axhline(0, c='k')


def get_visits(datadir: Path = datadir):
    return sorted(Path(datadir).glob('PR??????_TG??????_V????'))


def read_visit(visit: Path, kind: str = 'optimal', tw: bool = False):
    visit_int = int(visit.name.split('_')[1][-2:])
    try:
        lcfile = list(visit.glob(f'*Lightcurve*{kind.upper()}*.fits'))[0]
    except IndexError:
        print(f"Could not find '{kind}' light curve.")
        return None
    
    cvfile = list(visit.glob(f'*HkCe-SubArray*.fits'))[0]
    lcd = Table.read(lcfile).to_pandas()
    lcd.columns = [c.lower() for c in lcd.columns]
    lcd = lcd['bjd_time flux fluxerr status dark background conta_lc smearing_lc roll_angle centroid_x centroid_y'.split()]
    cvd = Table.read(cvfile).to_pandas()
    cvd.columns = [c.lower() for c in cvd.columns]
    df = pd.merge(lcd, cvd.iloc[:,11:], left_index=True, right_index=True)
    
    if tw:    
        lcfile2 = f'data/tom_wilson/Kelt-1_{visit_int:02d}.txt'
        df2 = pd.read_csv(lcfile2, delim_whitespace=True, header=None)
        df2.columns = ['bjd_time', 'flux', 'fluxerr']
   
        indices = []
        for t in df2.bjd_time.values:
            indices.append(argmin(abs(df.bjd_time.values - t)))
        indices = array(indices)

        df = df.iloc[indices, 3:].copy().reset_index()
        df.drop(columns='index', inplace=True)
        
        df = pd.merge(df2, df, left_index=True, right_index=True)

    return df


def clean_data(dfo):
    df = dfo.copy()
    df['sin_1ra'] = sin(df.roll_angle)
    df['cos_1ra'] = cos(df.roll_angle)
    df['sin_2ra'] = sin(2 * df.roll_angle)
    df['cos_2ra'] = cos(2 * df.roll_angle)

    mf = median_filter(df.flux, 5)

    mask_status = df.status == 0
    df.drop(columns=['status'], inplace=True)
    df = df.astype('d')
    
    mask_bckg = ~sigma_clip(df.background, sigma=5, stdfunc=mad_std).mask
    bckg_mf = median_filter(df.background[mask_bckg], 5)
    mask_bckg[mask_bckg] &= ~sigma_clip(df.background[mask_bckg] - bckg_mf, 3).mask

    mask_drdt = ones(df.shape[0], bool)
    drdt = diff(df.roll_angle) / diff(df.bjd_time)
    mask_drdt[1:] = ~sigma_clip(drdt, 10, stdfunc=mad_std).mask

    mask = mask_drdt & mask_bckg

    mf = median_filter(df.flux[mask], 5)
    mask[mask] &= ~sigma_clip(df.flux[mask] - mf, 4).mask

    time, flux = df.bjd_time.values[mask], df.flux.values[mask]
    flux /= median(flux)
    
    cov = df.drop(columns=['flux', 'fluxerr']).values[mask]
    cov = ((cov - cov.mean(0)) / cov.std(0))
    return time, flux, cov


def read_data(visits = None, tw: bool = False):
    visit_dirs = get_visits()
    if isinstance(visits, int):
        visits = [visits]
    visits = visits if visits is not None else range(1, len(visit_dirs) + 1)
    time, flux, cov = [], [], []
    for i in visits:
        df = read_visit(visit_dirs[i - 1], tw=tw)
        t, f, c = clean_data(df)
        time.append(t)
        flux.append(f)
        cov.append(c)
    return time, flux, cov


def read_tw_lightcurve(visit):
    files = sorted(Path('data/tom_wilson/').glob('*.dat'))
    df = pd.read_csv(files[visit-1]).dropna(axis=1)
    time, flux, cov = [df.BJD.values.copy()], [df.Flux.values.copy()], [df.iloc[:,3:].values.copy()]
    cov[0] = (cov[0] - cov[0].mean(0)) / cov[0].std(0)    
    return time, flux, cov


def load_detrended_cheops():
    df = Table.read('data/cheops_final_detrended.fits').to_pandas()
    times, fluxes, covs, pbs = [], [], [], []
    for v in df.visit.unique():
        m = df.visit == v
        times.append(df.time_bjd[m].values.copy())
        fluxes.append(df.flux_cor[m].values.copy())
        covs.append(array([[]]))
        pbs.append('CHEOPS')
    nids = arange(len(df.visit.unique()))
    return times, fluxes, covs, pbs, nids

