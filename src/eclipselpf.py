import seaborn as sb
from matplotlib.pyplot import subplots, setp
from numpy import inf, repeat, arctan2, atleast_2d, sqrt, squeeze, linspace, \
    median, percentile, argsort
from numpy.random.mtrand import permutation
from pytransit import EclipseModel, BaseLPF, LinearModelBaseline
from pytransit.lpf.loglikelihood import LogisticLogLikelihood
from pytransit.lpf.tesslpf import downsample_time
from pytransit.orbits.orbits_py import as_from_rhop, i_from_ba
from pytransit.param import GParameter, PParameter, NormalPrior as NP, UniformPrior as UP
from pytransit.param.prior import LaplacePrior
from pytransit.utils.misc import fold
from scipy.optimize import minimize
from scipy.stats import norm, truncnorm

from src.kelt1 import t0, period, rho, b, k2

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

        for p in self.ps[self._sl_lm]:
            p.prior = LaplacePrior(p.prior.mean, p.prior.std)

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

    def plot_folded_eclipse_model(self, method: str = 'optimization', nsamples: int = 300, binwidth: float = 0.01,
                                  ylim=None):
        assert method in ('optimization', 'mcmc')
        tm = EclipseModel()

        t0, p = self.de.minimum_location[[0, 1]]
        phasem = fold(self.timea, p, t0, normalize=False)
        time_m = linspace(t0 + phasem.min(), t0 + phasem.max(), 500)
        tm.set_data(time_m - self._tref)

        fo = self.ofluxa
        if method == 'optimization':
            pv = self.de.minimum_location
            fm = self.transit_model(pv, tm=tm)
            bl = self.baseline(pv)
        else:
            df = self.posterior_samples(derived_parameters=False)
            pvs = permutation(df.values)[:nsamples]
            pv = median(pvs, 0)
            fms = self.transit_model(pvs, tm=tm)
            bls = self.baseline(pvs)
            bl = median(bls, 0)
            fm, fmm, fmp = percentile(fms, [50, 2.5, 97.5], 0)

        t0, p = pv[[0, 1]]
        phaseo = fold(self.timea, p, t0, normalize=False)
        phasem = fold(time_m, p, t0, normalize=False)

        sids = argsort(phaseo)
        pb, fb, eb = downsample_time(phaseo[sids], (fo / bl)[sids], binwidth)

        fig, ax = subplots()
        ax.errorbar(24 * (pb - 0.5 * p), fb, eb, fmt='ok')
        for s in self.lcslices:
            ax.plot(24 * (phaseo[s] - 0.5 * p), fo[s] / bl[s], 'k.', alpha=0.1)
        if method == 'mcmc':
            ax.fill_between(24 * (phasem - 0.5 * p), fmm, fmp, alpha=0.5)
        ax.plot(24 * (phasem - 0.5 * p), fm, 'k')
        ax.autoscale(enable=True, axis='x', tight=True)
        setp(ax, xlabel='Time - T$_e$ [h]', ylabel='Normalised flux', ylim=ylim)
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


class LogisticEclipseLPF(EclipseLPF):

    def _init_lnlikelihood(self):
        self._add_lnlikelihood_model(LogisticLogLikelihood(self))