from pathlib import Path
from typing import Optional, List, Union

import pandas as pd
import astropy.io.fits as pf
from astropy.table import Table

from numpy import isfinite, diff, ones, median, c_, array, concatenate, sqrt, full
from numpy.polynomial.legendre import legvander
from pytransit.utils.downsample import downsample_time_1d, downsample_time_2d
#from pytransit.utils.tess import read_tess_spoc
from scipy.ndimage import label
from scipy.signal import medfilt

spitzer_data_dir = Path('data') / 'spitzer'

spitzer_files = ('KELT1_2012_36um_SptzLC.csv',
                 'KELT1_2015_36um_SptzLC.csv',
                 'KELT1_2012_45um_SptzLC.csv',
                 'KELT1_2015_45um_SptzLC.csv')


def downsample_data(times, fluxes, covs, bwidth=10.):
    btimes, bfluxes, bcovs = [], [], []
    for t, f, c in zip(times, fluxes, covs):
        tb, fb, eb = downsample_time_1d(t, f, bwidth / 60 / 24)
        _, cb, _ = downsample_time_2d(t, c, bwidth / 60 / 24)
        m = isfinite(tb)
        btimes.append(tb[m])
        bfluxes.append(fb[m])
        bcovs.append(cb[m])
    return btimes, bfluxes, bcovs


def load_spitzer(downsampling=None, nleg=1):
    """Imports the 3.6 um and 4.5 um Spitzer data from Beatty et al. (2014) and Beatty et al. (2019).
    """
    times, fluxes, covs, pbs = [], [], [], []
    for f in spitzer_files:
        df = pd.read_csv(spitzer_data_dir / f)
        time, flux, ferr = df.values[:, :3].T
        cov = df.values[:, -2:]
        cov = (cov - cov.mean(0)) / cov.std(0)
        pb = f.split('_')[2]

        c = 2 * (cov - cov.min(0)) / cov.ptp(0) - 1.0
        l = legvander(c, nleg)[:, :, 1:]
        ll = l.reshape([l.shape[0], -1])
        ll = (ll - ll.mean(0)) / ll.std(0)

        if '2015' in f:
            dy = diff(medfilt(cov[:, 1], 13))
            m = ones(time.size, 'bool')
            m[1:] = dy < 20 * dy.std()
            labels, nl = label(m)
            for i in range(1, nl + 1):
                lm = labels == i
                times.append(time[lm][100:-100].copy())
                fluxes.append(flux[lm][100:-100].copy())
                covs.append(ll[lm][100:-100].copy())
                pbs.append(pb)
        else:
            times.append(time)
            fluxes.append(flux)
            covs.append(ll)
            pbs.append(pb)

    if downsampling is not None:
        times, fluxes, covs = downsample_data(times, fluxes, covs, downsampling)
    return times, fluxes, covs, pbs


def load_beatty_2017(nleg=0, downsampling=None):
    """Imports the H-band data observed with the LBT from Beatty et al. (2017).
    """

    dfk = pd.read_csv('data/beatty/KELT1_H_Broad.dat', delim_whitespace=True)
    time = dfk.mjd.values + 2456590
    flux = (dfk.kelt1 / (dfk.comp1 + dfk.comp2)).values
    flux /= median(flux)
    covs = dfk.values[:, 5:]
    if nleg > 0:
        covs = c_[covs, legvander((time - time.min()) / time.ptp() * 2 - 1, nleg)[:, 1:]]
    covs = (covs - covs.mean(0)) / covs.std(0)

    times, fluxes, covs, pbs = [time], [flux], [covs], ['H']
    if downsampling is not None:
        times, fluxes, covs = downsample_data(times, fluxes, covs, downsampling)
    return times, fluxes, covs, pbs


def load_croll_2015(nleg=0, downsampling=None):
    """Imports the Ks-band data observed with the XX from Croll et al. (20XX).
    """

    df = pd.read_csv('data/croll/KELT1_Ksband_bbbnlcZZZxy2014NRSBFFITS_rap_19.00_hm_1_Nstar_4_typsel_2_finished_pre_analysis.txt',
           header=None, delim_whitespace=True)
    df.columns = 'bjd flux ferr x y'.split()
    time =  df.bjd.values.copy()
    flux = df.flux.values.copy()
    covs = df.values[:, 3:]
    if nleg > 0:
        covs = c_[covs, legvander((time - time.min()) / time.ptp() * 2 - 1, nleg)[:, 1:]]
    covs = (covs - covs.mean(0)) / covs.std(0)

    m = flux > 0.99
    times, fluxes, covs, pbs = [time[m]], [flux[m]], [covs[m]], ['Ks']
    if downsampling is not None:
        times, fluxes, covs = downsample_data(times, fluxes, covs, downsampling)
    return times, fluxes, covs, pbs


def read_tess_spoc(tic: int,
                   datadir: Union[Path, str],
                   sectors: Optional[Union[List[int], str]] = 'all',
                   use_pdc: bool = False,
                   remove_contamination: bool = True,
                   use_quality: bool = True):

    def file_filter(f, partial_tic, sectors):
        _, sector, tic, _, _ = f.name.split('-')
        if sectors != 'all':
            return int(sector[1:]) in sectors and str(partial_tic) in tic
        else:
            return str(partial_tic) in tic

    files = [f for f in sorted(Path(datadir).glob('tess*_lc.fits')) if file_filter(f, tic, sectors)]
    fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
    times, fluxes, sectors, quality = [], [], [], []
    for f in files:
        tb = Table.read(f)
        bjdrefi = tb.meta['BJDREFI']
        df = tb.to_pandas().dropna(subset=['TIME', fcol])
        q = df['QUALITY'].values.copy()
        m = (q == 0) if use_quality else ones(df.shape[0], bool)
        quality.append(q[m])
        times.append(df['TIME'].values[m].copy() + bjdrefi)
        fluxes.append(array(df[fcol].values[m], 'd'))
        fluxes[-1] /= median(fluxes[-1])
        if use_pdc and not remove_contamination:
            contamination = 1 - tb.meta['CROWDSAP']
            fluxes[-1] = contamination + (1 - contamination) * fluxes[-1]
        sectors.append(full(fluxes[-1].size, pf.getval(f, 'sector')))

    return (concatenate(times), concatenate(fluxes), concatenate(sectors), concatenate(quality),
            [diff(f).std() / sqrt(2) for f in fluxes])


def load_tess(downsampling=10.):
    times, fluxes, sectors, quality, wns = read_tess_spoc(432549364, 'data', sectors=[17], use_pdc=True, use_quality=True)
    if downsampling is not None:
        times, fluxes, _ = downsample_time_1d(times, fluxes, downsampling / 60 / 24)
        m = isfinite(times)
        times, fluxes = times[m], fluxes[m]
    return [times], [fluxes], [array([[]])], ['TESS']
