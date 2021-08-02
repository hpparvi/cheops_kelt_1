from matplotlib.pyplot import subplots, setp
from numpy import inf, repeat, pi, atleast_2d, arctan2, sqrt, squeeze, argsort, ones_like, isfinite
from pytransit import QuadraticModel, EclipseModel, NormalPrior as NP, UniformPrior as UP
from pytransit.lpf.lpf import map_ldc
from pytransit.lpf.tesslpf import TESSLPF, downsample_time
from pytransit.orbits import as_from_rhop, i_from_ba
from pytransit.param import GParameter, PParameter
from pytransit.utils.misc import fold

from .kelt1 import zero_epoch, period, tic
from .models import doppler_boosting, ellipsoidal_variation, emission, lambert_phase_function


class TESSPhaseCurveLPF(TESSLPF):
    def _post_initialisation(self):
        self.tm = QuadraticModel(interpolate=False)
        self.em = EclipseModel()
        self.tm.set_data(self.timea - self._tref, self.lcids, self.pbids, self.nsamples, self.exptimes)
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
        ldc = map_ldc(pv[:, self._sl_ld])
        aev = pv[:, 7]
        adb = pv[:, 8]
        dte = pv[:, 9]
        nte = pv[:, 10]
        ote = pv[:, 11]
        ag = pv[:, 12]
        return t0, p, a, inc, ecc, omega, area_ratio, k, ldc, aev, adb, dte, nte, ote, ag

    def baseline(self, pv):
        pv = atleast_2d(pv)
        return squeeze(pv[:, self._ix_bl])

    def doppler_boosting(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        return squeeze(doppler_boosting(adb, t0, p, True, self.timea))

    def ellipsoidal_variations(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        return squeeze(ellipsoidal_variation(aev, t0, p, True, self.timea))

    def thermal_flux(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        ft = emission(area_ratio, nte, dte, ote, t0, p, True, self.timea)
        fec = self.em.evaluate(k, t0, p, a, inc, ecc, omega, multiplicative=True)
        return squeeze(fec*ft)

    def reflected_flux(self, pv):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        fr = lambert_phase_function(a, area_ratio, ab, t0, p, True, self.timea)
        fec = self.em.evaluate(k, t0, p, a, inc, ecc, omega, multiplicative=True)
        return squeeze(fec*fr)

    def transit_model(self, pv, copy=True):
        t0, p, a, inc, ecc, omega, area_ratio, k, ldc, aev, adb, dte, nte, ote, ab = self.map_pv(pv)
        fec = self.em.evaluate(k, t0 - self._tref, p, a, inc, ecc, omega, multiplicative=True)
        ft = emission(area_ratio, nte, dte, ote, t0, p, True, self.timea)
        fr = lambert_phase_function(a, area_ratio, ab, t0, p, True, self.timea)
        fplanet = fec * squeeze((ft + fr))

        ftr = self.tm.evaluate(k, ldc, t0 - self._tref, p, a, inc, ecc, omega)
        fev = ellipsoidal_variation(aev, t0, p, True, self.timea)
        fdb = doppler_boosting(adb, t0, p, True, self.timea)
        fstar = squeeze(ftr + fev + fdb) #ev_and_db(aev, adb, t0, p, True, self.timea)
        return fplanet + fstar

    def flux_model(self, pv):
        baseline = self.baseline(pv)
        model_flux = self.transit_model(pv)
        return squeeze(model_flux * atleast_2d(baseline).T)

    def plot_folded_transit(self, method='de', figsize=(13, 6), ylim=(0.9975, 1.002), xlim=None, binwidth=8,
                            remove_baseline: bool = False, offset: float = 0.5):
        if method == 'de':
            pv = self.de.minimum_location
            tc, p = pv[[0, 1]]
        else:
            raise NotImplementedError

        phase = p * fold(self.timea, p, tc, offset)
        binwidth = binwidth / 24 / 60
        sids = argsort(phase)

        tm = self.transit_model(pv)

        if remove_baseline:
            gp = self._lnlikelihood_models[0]
            bl = squeeze(gp.predict_baseline(pv))
        else:
            bl = ones_like(self.ofluxa)

        bp, bfo, beo = downsample_time(phase[sids], (self.ofluxa / bl)[sids], binwidth)
        _, bfm, _ = downsample_time(phase[sids], tm[sids], binwidth)

        fig, ax = subplots(figsize=figsize)
        ax.plot(phase - offset * p, self.ofluxa / bl, '.', alpha=0.05)
        ax.errorbar(bp - offset * p, bfo, beo, ms=12, fmt='wo', alpha=0.85)
        ax.errorbar(bp - offset * p, bfo, beo, fmt='ko')
        ax.errorbar(bp - offset * p, bfo - bfm + 0.9915, beo, fmt='ko')
        ax.plot(phase[sids] - offset * p, tm[sids], 'k', alpha=1, zorder=10)
        ax.axvline(0.0, c='k', ls="--")
        ax.axvline(0.5 * p, c='k', ls=":")
        xlim = xlim if xlim is not None else 1.01 * (bp[isfinite(bp)][[0, -1]] - offset * p)
        setp(ax, ylim=ylim, xlim=xlim, xlabel='Time - Tc [d]', ylabel='Normalised flux')
        fig.tight_layout()
        return fig

class K1TESSLPF(TESSPhaseCurveLPF):
    def __init__(self, run_name: str = ''):
        super().__init__('KELT-1b-TESS', 'data', tic, zero_epoch.n, period.n, trdur=0.2, bldur=2.0)
        if run_name:
            self.name = f"{self.name}-{run_name}"

    def _post_initialisation(self):
        super()._post_initialisation()

        # PRIORS
        # ------
        # I use relatively uninformative priors on most of the model parameters. The priors mainly make sure that
        # the optimizer has a sensible starting population to begin the optimization.

        self.set_prior('tc', 'NP', zero_epoch.n, 0.05) # - Zero epoch: normal prior with an inflated uncertainty
        self.set_prior('p', 'NP', period.n, 1e-3)      # - Orbital period: normal prior with an inflated uncertainty
        self.set_prior('rho', 'UP', 0.1, 1.0)          # - Stellar density: wide uniform prior
        self.set_prior('secw', 'NP', 0.0, 1e-6)        # - Circular orbit: sqrt(e) cos(w) and sqrt(e) sin(w) forced
        self.set_prior('sesw', 'NP', 0.0, 1e-6)        #   to zero with normal priors.
        self.set_prior('q1_TESS', 'NP', 0.27, 0.01)    # - Quadratic (triangular) limb darkening coefficients
        self.set_prior('q2_TESS', 'NP', 0.34, 0.01)    #   estimated using LDTk (with inflated uncertainties).
        self.set_prior('aev', 'UP', 0.0, 600e-6)       # - Ellipsoidal variation
        self.set_prior('adb', 'NP', 41e-6, 3e-6)       # - Doppler boosting amplitude
        self.set_prior('ted', 'UP', 0.0, 0.2)          # - Emission day-side flux ratio
        self.set_prior('ten', 'UP', 0.0, 0.1)          # - Emission night-side flux ratio
        self.set_prior('teo', 'UP', -0.5*pi, 0.5*pi)   # - Emission offset
        self.set_prior('ag', 'UP', 0.0, 1.0)           # - Geometric albedo
        self.set_prior('f0', 'NP', 1.0, 0.002)         # - Baseline flux level: wide normal prior
        self.set_radius_ratio_prior(0.05, 0.15)

