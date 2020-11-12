from pathlib import Path

import pandas as pd
from matplotlib.pyplot import subplots, setp
from numpy import radians, sin, cos, diff, inf, repeat, arctan2, atleast_2d, sqrt, squeeze, ones, argsort, linspace, \
    median, percentile
from numpy.random.mtrand import permutation
from scipy.ndimage import median_filter

from astropy.stats import sigma_clip, mad_std
from pytransit import EclipseModel, BaseLPF, LinearModelBaseline
from pytransit.lpf.loglikelihood import CeleriteLogLikelihood
from pytransit.param import GParameter, PParameter, NormalPrior as NP, UniformPrior as UP
from pytransit.orbits.orbits_py import as_from_rhop, i_from_ba, epoch
from pytransit.utils.misc import fold
from scipy.optimize import minimize
from uncertainties import ufloat

import seaborn as sb
from scipy.stats import norm, truncnorm

# Planetary parameters
# --------------------

t0 = ufloat(2458764.31647, 0.00018)
period = ufloat(1.2174928, 1.7e-6)
rho = ufloat(0.58, 0.05)
b = ufloat(0.195, 0.05)
k2 = ufloat(0.005935, 4.468e-5)

# Utility functions
# -----------------

def get_data_files(datadir: Path = 'data'):
    return sorted(Path(datadir).glob('*.rdb'))


def read_data_file(fname):
    df = pd.read_csv(fname, sep='\t', )
    df = df.iloc[3:]
    df = df['obj_date_bjd_vect photom_flux_vect photom_flux_vect_err photom_centroid_x_vect photom_centroid_y_vect photom_roll_angle_vect photom_background_vect'.split()]
    df.columns = 'time flux flux_err centroid_x centroid_y roll_angle background'.split()
    df = df.astype('d')
    df.time += 2400000
    df.roll_angle = radians(df.roll_angle)
    return df


def clean_data(dfo):
    df = dfo.copy()
    df['sin_1ra'] = sin(df.roll_angle)
    df['cos_1ra'] = cos(df.roll_angle)
    df['sin_2ra'] = sin(2 * df.roll_angle)
    df['cos_2ra'] = cos(2 * df.roll_angle)

    mf = median_filter(df.flux, 5)

    mask_bckg = ~sigma_clip(df.background, sigma=5, stdfunc=mad_std).mask
    bckg_mf = median_filter(df.background[mask_bckg], 5)
    mask_bckg[mask_bckg] &= ~sigma_clip(df.background[mask_bckg] - bckg_mf, 3).mask

    mask_drdt = ones(df.shape[0], bool)
    drdt = diff(df.roll_angle) / diff(df.time)
    mask_drdt[1:] = ~sigma_clip(drdt, 10, stdfunc=mad_std).mask

    mask = mask_drdt & mask_bckg

    mf = median_filter(df.flux[mask], 5)
    mask[mask] &= ~sigma_clip(df.flux[mask] - mf, 4).mask

    time, flux = df.time.values[mask], df.flux.values[mask]

    cov = df.drop(columns=['flux', 'flux_err']).values[mask]
    cov = ((cov - cov.mean(0)) / cov.std(0))
    return time, flux, cov


def read_data(visits=None):
    data_files = get_data_files()
    if isinstance(visits, int):
        visits = [visits]
    visits = visits if visits is not None else range(1, len(data_files) + 1)
    time, flux, cov = [], [], []
    for i in visits:
        df = read_data_file(data_files[i - 1])
        t, f, c = clean_data(df)
        time.append(t)
        flux.append(f)
        cov.append(c)
    return time, flux, cov


# The log posterior function
# --------------------------

class EclipseLPF(BaseLPF):

    def _init_p_limb_darkening(self):
        pass

    def _init_p_orbit(self):
        porbit = [
            GParameter('tc', 'zero epoch', 'd', NP(0.0, 0.1), (-inf, inf)),
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('rho', 'stellar density', 'g/cm^3', UP(0.1, 25.0), (0, inf)),
            GParameter('b', 'impact parameter', 'R_s', UP(0.0, 1.0), (0, 1)),
            GParameter('secw', 'sqrt(e) cos(w)', '', UP(-0.5, 0.5), (-1, 1)),
            GParameter('sesw', 'sqrt(e) sin(w)', '', UP(-0.5, 0.5), (-1, 1))]
        self.ps.add_global_block('orbit', porbit)

    def _init_p_planet(self):
        """Planet parameter initialisation.
        """
        pk2 = [PParameter('k2', 'area ratio', 'A_s', UP(0.01 ** 2, 0.2 ** 2), (0, inf))]
        self.ps.add_global_block('k2', pk2)
        self._pid_k2 = repeat(self.ps.blocks[-1].start, 1)
        self._start_k2 = self.ps.blocks[-1].start
        self._sl_k2 = self.ps.blocks[-1].slice
        self._ix_k2 = self._start_k2

        pfr = [PParameter(f'fr_{pb}', 'Flux ratio', '', UP(0, 1), (0, 1)) for pb in self.passbands]
        self.ps.add_passband_block('fr', len(self.passbands), 1, pfr)
        self._pid_fr = repeat(self.ps.blocks[-1].start, self.npb)
        self._start_fr = self.ps.blocks[-1].start
        self._sl_fr = self.ps.blocks[-1].slice

    # def _init_lnlikelihood(self):
    #    self._add_lnlikelihood_model(FrozenMultiCeleriteLogLikelihood(self))

    #def _init_lnlikelihood(self):
    #   self._add_lnlikelihood_model(CeleriteLogLikelihood(self))

    def _init_baseline(self):
        self._add_baseline_model(LinearModelBaseline(self))

    def _post_initialisation(self):
        super()._post_initialisation()
        self.tm = EclipseModel()
        self.tm.set_data(self.timea - self._tref, self.lcids, self.pbids, self.nsamples, self.exptimes)

        self.set_prior('tc', 'NP', t0.n, t0.s)
        self.set_prior('p', 'NP', period.n, period.s)
        self.set_prior('rho', 'NP', rho.n, rho.s)
        self.set_prior('b', 'NP', b.n, b.s)
        self.set_prior('secw', 'NP', 0.0, 1e-4)
        self.set_prior('sesw', 'NP', 0.0, 1e-4)
        self.set_prior('k2', 'NP', k2.n, k2.s)
        self.set_prior('fr_cheops', 'UP', 0.0, 0.2)

    def transit_model(self, pv, copy=True, tm=None):
        tm = tm if tm is not None else self.tm
        pv = atleast_2d(pv)
        zero_epoch = pv[:, 0] - self._tref
        period = pv[:, 1]
        smaxis = as_from_rhop(pv[:, 2], period)
        inclination = i_from_ba(pv[:, 3], smaxis)
        radius_ratio = sqrt(pv[:, 6:7])
        eccentricity = pv[:, 4] ** 2 + pv[:, 5] ** 2
        omega = arctan2(pv[:, 5], pv[:, 4])
        fratio = pv[:, 7:8]
        fmod = tm.evaluate(radius_ratio, zero_epoch, period, smaxis, inclination, eccentricity, omega, fratio)
        return squeeze(fmod)

    def create_pv_population(self, npop):
        return self.ps.sample_from_prior(npop)

    def plot_folded_eclipse_model(self, method: str = 'optimization', nsamples: int = 300):
        assert method in ('optimization', 'mcmc')
        tm = EclipseModel()

        time_m = linspace(self.times[0][0], self.times[0][-1], 500)
        tm.set_data(time_m - self._tref)

        phaseo = fold(self.timea, period.n, t0.n, normalize=False)
        phasem = fold(time_m, period.n, t0.n, normalize=False)

        fo = self.ofluxa
        if method == 'optimization':
            fm = self.transit_model(self.de.minimum_location, tm=tm)
            bl = self.baseline(self.de.minimum_location)
        else:
            df = self.posterior_samples(derived_parameters=False)
            pvs = permutation(df.values)[:nsamples]
            fms = self.transit_model(pvs, tm=tm)
            bls = self.baseline(pvs)
            bl = median(bls, 0)
            fm, fmm, fmp = percentile(fms, [50, 2.5, 97.5], 0)

        fig, ax = subplots()
        for s in self.lcslices:
            ax.plot(phaseo[s], fo[s] / bl[s], '.', alpha=0.5)
        if method == 'mcmc':
            ax.fill_between(phasem, fmm, fmp, alpha=0.5)
        ax.plot(phasem, fm, 'k')
        fig.tight_layout()
        return fig

    def plot_flux_model(self, method: str = 'optimization', nsamples: int = 300):
        assert method in ('optimization', 'mcmc')

        time = self.timea - self._tref

        fo = self.ofluxa
        if method == 'optimization':
            fm = self.flux_model(self.de.minimum_location)
        else:
            df = self.posterior_samples(derived_parameters=False)
            pvs = permutation(df.values)[:nsamples]
            fm, fmm, fmp = percentile(self.flux_model(pvs), [50, 2.5, 97.5], 0)

        fig, ax = subplots()
        for s in self.lcslices:
            ax.plot(time[s], fo[s], '.', alpha=0.5)
        if method == 'mcmc':
            ax.fill_between(time, fmm, fmp, alpha=0.5)
        ax.plot(time, fm, 'k')
        setp(ax, xlabel=f'Time - {self._tref:.0f} [BJD]', ylabel='Normalized flux')
        fig.tight_layout()
        return fig

    def plot_posteriors(self, bins: int = 40, figsize=None, truncate: bool = False, nsamples: int = 1000):
        df = self.posterior_samples(derived_parameters=False)

        fig, axs = subplots(1, 2, figsize=figsize)
        for i, p in enumerate((df.fr_cheops, 1e6 * df.fr_cheops * df.k2)):
            axs[i].hist(p, bins=bins, density=True)

            if truncate:

                def neglln(x, samples, vmax):
                    m, s = x
                    a, b = 0.0, vmax
                    return -truncnorm((a - m) / s, (b - m) / s, m, s).logpdf(samples).sum()

                samples = permutation(p)[:nsamples]
                vmax = samples.max()
                res = minimize(neglln, (median(samples), samples.std()), (samples, vmax))
                mm, ms = res.x
                a = -mm / ms
                b = (vmax - mm) / ms
                x = linspace(max(0.0, mm - 5 * ms), mm + 5 * ms)

                axs[i].plot(x, truncnorm.pdf(x, a, b, mm, ms), 'k')
                axs[i].plot([mm, mm], [0, truncnorm.pdf(mm, a, b, mm, ms)], 'k--', lw=1.5)
            else:
                mm, ms = p.mean(), p.std()
                x = linspace(max(0.0, mm - 5 * ms), mm + 5 * ms)
                axs[i].plot(x, norm.pdf(x, mm, ms), 'k')
                axs[i].plot([mm, mm], [0, norm.pdf(mm, mm, ms)], 'k--', lw=1.5)
            axs[i].text(0.98, 0.9, f"$\mu=${mm:.3f}\n$\sigma=${ms:.3f}", ha='right', transform=axs[i].transAxes)

        setp(axs[0], xlabel='Planet-star flux ratio', ylabel='Posterior probability', yticks=[], xlim=(0.0, 0.12))
        setp(axs[1], xlabel='Eclipse depth [ppm]', yticks=[], xlim=(0, 700))
        sb.despine(fig, offset=8)
        fig.tight_layout()
        return fig