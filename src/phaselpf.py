import seaborn as sb

from matplotlib.pyplot import subplots, setp
from numpy import inf, repeat, arctan2, atleast_2d, sqrt, squeeze, pi, median, percentile, linspace
from numpy.random import permutation
from pytransit import EclipseModel
from pytransit.orbits.orbits_py import as_from_rhop, i_from_ba
from pytransit.param import GParameter, PParameter, NormalPrior as NP, UniformPrior as UP
from pytransit.utils.misc import fold
from scipy.optimize import minimize
from scipy.stats import truncnorm, norm

from .eclipselpf import EclipseLPF
from .models import ellipsoidal_variation, lambert_phase_function, doppler_boosting, emission


class PhaseLPF(EclipseLPF):
    def _post_initialisation(self):
        self.em = EclipseModel()
        self.em.set_data(self.timea - self._tref, self.lcids, self.pbids, self.nsamples, self.exptimes)

    def _init_p_orbit(self):
        porbit = [
            GParameter('tc', 'zero epoch', 'd', NP(0.0, 0.1), (-inf, inf)),
            GParameter('p', 'p', 'd', NP(1.0, 1e-5), (0, inf)),
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

        pph = [GParameter('aev', 'Ellipsoidal variation amplitude', '', UP(0, 1), (0, inf)),
               GParameter('adb', 'Doppler boosting amplitude', '', UP(0, 1), (0, inf)),
               GParameter('ted', 'Day-side flux ratio', '', UP(0.0, 0.2), (-inf, inf)),
               GParameter('ten', 'Night-side flux ratio', '', UP(0.0, 0.1), (-inf, inf)),
               #GParameter('ted', 'Day-side flux ratio', '', UP(-3, 0), (-inf, inf)),
               #GParameter('ten', 'Night-side flux ratio', '', UP(-3, 0), (-inf, inf)),
               GParameter('teo', 'Thermal emission offset', 'rad', UP(-pi, pi), (-inf, inf)),
               GParameter('ag', 'Geometric albedo', '', UP(0, 1), (0, 1))]
        self.ps.add_global_block('phase', pph)
        self._pid_fr = repeat(self.ps.blocks[-1].start, 1)
        self._start_fr = self.ps.blocks[-1].start
        self._sl_fr = self.ps.blocks[-1].slice

    def _init_p_baseline(self):
        pbl = [PParameter('f0', 'baseline flux', '', UP(0.99, 1.01), (0, inf))]
        self.ps.add_global_block('bl', pbl)
        self._pid_bl = repeat(self.ps.blocks[-1].start, 1)
        self._start_bl = self.ps.blocks[-1].start
        self._sl_bl = self.ps.blocks[-1].slice
        self._ix_bl = self._start_bl

    def map_pv(self, pv):
        pv = atleast_2d(pv)
        t0 = pv[:, 0]
        p = pv[:, 1]
        a = as_from_rhop(pv[:, 2], p)
        inc = i_from_ba(pv[:, 3], a)
        ecc = pv[:, 4] ** 2 + pv[:, 5] ** 2
        omega = arctan2(pv[:, 5], pv[:, 4])
        area_ratio = pv[:, 6]
        k = sqrt(pv[:, 6:7])
        aev = pv[:, 7]
        adb = pv[:, 8]
        #dte = 10**pv[:, 9]
        #nte = 10**pv[:, 10]
        dte = pv[:, 9]
        nte = pv[:, 10]
        ote = pv[:, 11]
        ag = pv[:, 12]
        return t0, p, a, inc, ecc, omega, area_ratio, k, aev, adb, dte, nte, ote, ag

    def doppler_boosting(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        return squeeze(doppler_boosting(adb, t0, p, True, self.timea))

    def ellipsoidal_variations(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        return squeeze(ellipsoidal_variation(aev, t0, p, True, self.timea))

    def thermal_flux(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        ft = emission(area_ratio, nte, dte, ote, t0, p, True, self.timea)
        fec = self.em.evaluate(k, t0, p, a, inc, ecc, omega, multiplicative=True)
        return squeeze(fec*ft)

    def reflected_flux(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        fr = lambert_phase_function(a, area_ratio, ab, t0, p, True, self.timea)
        fec = self.em.evaluate(k, t0, p, a, inc, ecc, omega, multiplicative=True)
        return squeeze(fec*fr)

    def transit_model(self, pv, copy=True, tm=None, time=None):
        tm = tm if tm is not None else self.em
        time = time if time is not None else self.timea
        t0, p, a, inc, ecc, omega, area_ratio, k, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        fec = tm.evaluate(k, t0 - self._tref, p, a, inc, ecc, omega, multiplicative=True)
        ft = emission(area_ratio, nte, dte, ote, t0, p, True, time)
        fr = lambert_phase_function(a, area_ratio, ab, t0, p, True, time)
        fplanet = fec * squeeze((ft + fr))

        fev = ellipsoidal_variation(aev, t0, p, True, time)
        fdb = doppler_boosting(adb, t0, p, True, time)
        fstar = 1.0 + squeeze(fev + fdb)
        return fplanet + fstar

    def flux_model(self, pv):
        baseline = self.baseline(pv)
        model_flux = self.transit_model(pv)
        return squeeze(model_flux * atleast_2d(baseline))

    def plot_folded_eclipse_model(self, method: str = 'optimization', nsamples: int = 300):
        assert method in ('optimization', 'mcmc')
        tm = EclipseModel()

        time_m = linspace(self.times[0][0], self.times[0][-1], 500)
        tm.set_data(time_m - self._tref)

        fo = self.ofluxa
        if method == 'optimization':
            pv = self.de.minimum_location
            fm = self.transit_model(pv, tm=tm, time=time_m)
            bl = self.baseline(pv)
        else:
            df = self.posterior_samples(derived_parameters=False)
            pvs = permutation(df.values)[:nsamples]
            pv = median(pvs, 0)
            fms = self.transit_model(pvs, tm=tm, time=time_m)
            bls = self.baseline(pvs)
            bl = median(bls, 0)
            fm, fmm, fmp = percentile(fms, [50, 2.5, 97.5], 0)

        t0, p = pv[[0, 1]]

        phaseo = fold(self.timea, p, t0, normalize=False)
        phasem = fold(time_m, p, t0, normalize=False)

        fig, ax = subplots()
        for s in self.lcslices:
            ax.plot(phaseo[s], fo[s]/bl[s], '.', alpha=0.5)
        if method == 'mcmc':
            ax.fill_between(phasem, fmm, fmp, alpha=0.5)
        ax.plot(phasem, fm, 'k')
        fig.tight_layout()
        return fig

    def plot_posteriors(self, bins: int = 40, figsize=None, truncate: bool = False, nsamples: int = 1000):
        df = self.posterior_samples(derived_parameters=False)

        fig, axs = subplots(1, 2, figsize=figsize)
        for i, p in enumerate((10**df.ted, 1e6*10**df.ted*df.k2)):
            axs[i].hist(p, bins=bins, density=True)

            if truncate:

                def neglln(x, samples, vmax):
                    m, s = x
                    a, b = 0.0, vmax
                    return -truncnorm((a - m)/s, (b - m)/s, m, s).logpdf(samples).sum()

                samples = permutation(p)[:nsamples]
                vmax = samples.max()
                res = minimize(neglln, (median(samples), samples.std()), (samples, vmax))
                mm, ms = res.x
                a = -mm/ms
                b = (vmax - mm)/ms
                x = linspace(max(0.0, mm - 5*ms), mm + 5*ms)

                axs[i].plot(x, truncnorm.pdf(x, a, b, mm, ms), 'k')
                axs[i].plot([mm, mm], [0, truncnorm.pdf(mm, a, b, mm, ms)], 'k--', lw=1.5)
            else:
                mm, ms = p.mean(), p.std()
                x = linspace(max(0.0, mm - 5*ms), mm + 5*ms)
                axs[i].plot(x, norm.pdf(x, mm, ms), 'k')
                axs[i].plot([mm, mm], [0, norm.pdf(mm, mm, ms)], 'k--', lw=1.5)
            axs[i].text(0.98, 0.9, f"$\mu=${mm:.3f}\n$\sigma=${ms:.3f}", ha='right', transform=axs[i].transAxes)

        setp(axs[0], xlabel='Planet-star flux ratio', ylabel='Posterior probability', yticks=[], xlim=(0.0, 0.12))
        setp(axs[1], xlabel='Eclipse depth [ppm]', yticks=[], xlim=(0, 700))
        sb.despine(fig, offset=8)
        fig.tight_layout()
        return fig