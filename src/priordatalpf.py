import pandas as pd
import seaborn as sb

from pathlib import Path

from matplotlib.pyplot import setp, subplots
from numpy import radians, argsort, percentile, median, array, ones, zeros, arange, concatenate
from numpy.random import permutation

from pytransit import LinearModelBaseline, NormalPrior as NP
from pytransit.lpf.loglikelihood import WNLogLikelihood, CeleriteLogLikelihood
from pytransit.lpf.phasecurvelpf import PhaseCurveLPF
from pytransit.utils.downsample import downsample_time_1d
from pytransit.orbits import epoch, fold

from src.dataimport import load_tess, load_beatty_2017, load_spitzer, load_croll_2015
from src.kelt1 import (zero_epoch, period, filter_names, beaming_amplitudes as ba, beaming_uncertainties as be,
                       ev_amplitudes as eva,)

ds = dict(lbt=2., croll=2., spitzer=2., tess=None)

class JointLPF(PhaseCurveLPF):
    def __init__(self, scenario: str, savedir: Path = Path('results'), downsampling = None):
        """KELT-1b joint model LPF.

        This LPF models the TESS, LBT, and Spitzer data jointly.

        Scenarios:
          a) Zero albedo and unconstrained emission, EV amplitude constrained by a theoretical prior.
          b) Zero albedo and unconstrained emission, EV amplitude constrained by a ratio prior.
          c) Zero albedo and unconstarined emission, EV amplitude unconstrained
          d) Unconstrained emission and reflection + constrained ev
        """
        self.scenarios = 'a b c d'.split()
        self.labels = {'a': 'emission_and theoretical_ev',
                       'b': 'emission_and_constrained_ev',
                       'c': 'emission_and_unconstrained_ev',
                       'd': 'emission_and_reflection_and_theoretical_ev'}

        if scenario not in self.scenarios:
            raise ValueError(f'The JointLPF scenario has to be one of {self.scenarios}')
        self.scenario = scenario

        if downsampling is not None:
            ds.update(downsampling)

        # Load in the data
        # ----------------
        tt, tf, tcov, tpb, tnids = load_tess(downsampling=ds['tess'])
        ht, hf, hcov, hpb, hnids = load_beatty_2017(downsampling=ds['lbt'])
        kt, kf, kcov, kpb, knids = load_croll_2015(downsampling=ds['croll'])
        st, sf, scov, spb, snids = load_spitzer(downsampling=ds['spitzer'], nleg=3)

        # Fix the noise IDs
        # -----------------
        hnids += 1+tnids[-1]
        knids += 1+hnids[-1]
        snids += 1+knids[-1]

        times = tt + ht + kt + st
        fluxes = tf + hf + kf + sf
        covs = tcov + hcov + kcov + scov
        pbs = pd.Categorical(tpb + hpb + kpb + spb, categories=['TESS', 'H', 'Ks', '36um', '45um'])
        noise_ids = concatenate([tnids, hnids, knids, snids]) #arange(len(times))

        # Initialise the PhaseCurveLPF and add in a linear baseline model
        # ---------------------------------------------------------------
        name = f"01{self.scenario}_ext_{self.labels[self.scenario]}"
        super().__init__(name, pbs.categories.values, times, fluxes, pbids=pbs.codes, covariates=covs,
                         wnids=noise_ids, result_dir=savedir)
        self._add_baseline_model(LinearModelBaseline(self))

    def _post_initialisation(self):
        super()._post_initialisation()
        self.set_prior('tc', 'NP', zero_epoch.n, 0.01)  # - Zero epoch: normal prior with an inflated uncertainty
        self.set_prior('p', 'NP', period.n, 3*period.s)       # - Orbital period: normal prior with an inflated uncertainty
        self.set_prior('rho', 'NP', 0.6, 0.075)         # - Stellar density: wide normal prior
        self.set_prior('secw', 'NP', 0.0, 1e-6)         # - Circular orbit: sqrt(e) cos(w) and sqrt(e) sin(w) forced
        self.set_prior('sesw', 'NP', 0.0, 1e-6)         #   to zero with normal priors.
        self.set_prior('k2', 'NP', 0.078, 0.005)        # - Area ratio: wide normal prior based on Siverd et al. (2012)

        self.set_prior('q1_36um', 'NP', 0.03, 0.0005)
        self.set_prior('q2_36um', 'NP', 0.28, 0.0040)
        self.set_prior('q1_45um', 'NP', 0.02, 0.0004)
        self.set_prior('q2_45um', 'NP', 0.29, 0.0050)

        self.set_prior('q1_H',  'NP', 0.5, 0.001)
        self.set_prior('q2_H',  'NP', 0.5, 0.001)
        self.set_prior('q1_Ks', 'NP', 0.5, 0.001)
        self.set_prior('q2_Ks', 'NP', 0.5, 0.001)

        # Set the default phase curve priors
        # ----------------------------------
        # We may modify these later based on the chosen
        # simulation scenario.
        self.set_prior('oev', 'NP', 0.0, 1e-7)
        for pb in self.passbands:
            self.set_prior(f'aev_{pb}', 'UP', 0.0, 1000e-6)      # - Ellipsoidal variation
            self.set_prior(f'ted_{pb}', 'UP', 0.0, 0.5)          # - Emission day-side flux ratio
            self.set_prior(f'ten_{pb}', 'UP', 0.0, 0.5)          # - Emission night-side flux ratio
            self.set_prior(f'teo_{pb}', 'NP', 0.0, radians(10))  # - Emission peak offset

        # Force the H and Ks band emission offset to zero
        # -----------------------------------------------
        # We do this because we don't have enough phase
        # coverage to measure the offsets.
        self.set_prior('teo_H', 'NP', 0.0, 1e-5)
        self.set_prior('teo_Ks', 'NP', 0.0, 1e-5)

        # Set priors on the Doppler beaming amplitudes
        # --------------------------------------------
        # The Doppler beaming amplitudes are estimated from
        # BT-Settl model spectra for KELT-1 (see Appendix 2
        # notebook), and vary from 65 ppm in the CHEOPS passband
        # to 15 ppm in the Spitzer passbands.
        self.set_prior('adb_TESS', 'NP', ba['TESS'], 2*be['TESS'])
        self.set_prior('adb_H',    'NP', ba['H'],    2*be['H'])
        self.set_prior('adb_Ks',   'NP', ba['Ks'],   2*be['Ks'])
        self.set_prior('adb_36um', 'NP', ba['36um'],   2*be['36um'])
        self.set_prior('adb_45um', 'NP', ba['45um'],   2*be['45um'])

        # Set a prior on the geometric albedo
        # -----------------------------------
        if self.scenario in 'abc':
            for pb in self.passbands:
                self.set_prior(f'ag_{pb}', 'NP', 1e-4, 1e-6)
        elif self.scenario == 'd':
            for pb in self.passbands:
                self.set_prior(f'ag_{pb}', 'UP', 0.0, 2.0)

        # Set a prior on ellipsoidal variation
        # ------------------------------------
        # This prior dictates the ratio between
        # the ev amplitude in passband x and the
        # reference (TESS) passband. This ratio can
        # be estimated theoretically, as is done in
        # the "xxx" notebook.
        if self.scenario in 'ad':
            for pb in self.passbands:
                self.set_prior(f'aev_{pb}', 'NP', eva[pb].n, eva[pb].s)
        elif self.scenario == "b":
            pr_e_h = NP(0.81, 0.05)
            pr_e_ks = NP(0.80, 0.05)
            pr_e_s1 = NP(0.78, 0.10)
            pr_e_s2 = NP(0.78, 0.10)
            def ev_amplitude_prior(pvp):
                h_ratio = pvp[:, 14] / pvp[:, 8]
                ks_ratio = pvp[:, 20] / pvp[:, 8]
                s1_ratio = pvp[:, 26] / pvp[:, 8]
                s2_ratio = pvp[:, 32] / pvp[:, 8]
                return pr_e_h.logpdf(h_ratio) + pr_e_ks.logpdf(ks_ratio) + pr_e_s1.logpdf(s1_ratio) + pr_e_s2.logpdf(s2_ratio)
            self.add_prior(ev_amplitude_prior)
        elif self.scenario == 'c':
            pass

    # Define the log likelihood
    # -------------------------
    # We use a GP log likelihood for the TESS, LBT, and CFHT data and
    # uncorrelated normal log likelihood for Spitzer else.
    def _init_lnlikelihood(self):
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[0]))  # TESS
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[1]))  # LBT H
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[2]))  # CFHT Ks
            self._add_lnlikelihood_model(WNLogLikelihood(self, noise_ids=array([3,4,5,6])))

def plot_joint_lcs(lpf, method='de', pv=None):
    yoff = 5e-3
    yoffsets = (yoff * array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0])).cumsum()
    colids = [0, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6]
    poffset = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    if pv is None:
        if method == 'de':
            pv = lpf.de.minimum_location.copy()
        else:
            pv = lpf.posterior_samples(derived_parameters=False)

    if pv.ndim == 1:
        t0, p = pv[[0, 1]]
    else:
        t0, p = median(pv[:, :2], 0)

    fig, ax = subplots(figsize=(13, 12))

    if pv.ndim == 1:
        fm = lpf.transit_model(pv)
        fmem, fmep = None, None
        bl = lpf.baseline(pv)
    else:
        fp = percentile(lpf.transit_model(permutation(pv)[:100]), [50, 16, 84], axis=0)
        fm, fmem, fmep = fp
        bl = median(lpf.baseline(permutation(pv)[:100]), axis=0)

    # TESS
    sl = lpf.lcslices[0]
    phase = fold(lpf.timea[sl], p, t0 + 0.18) + 0.18
    sids = argsort(phase)
    bp, bf, be = downsample_time_1d(phase[sids], (lpf.ofluxa[sl] / bl[sl])[sids], 10 / 60 / 24)
    ax.plot(bp, bf + yoffsets[0], '.', c="C1", zorder=-1)
    ax.plot(phase[sids], fm[sl][sids] + yoffsets[0], 'k-')

    # LBT and Spitzer
    for i, sl in enumerate(lpf.lcslices[1:]):
        ep = epoch(lpf.timea[sl].mean(), t0, p)
        tc = t0 + ep * p
        ax.plot(lpf.timea[sl] - tc + poffset[i] * p, lpf.ofluxa[sl] / bl[sl] + yoffsets[i + 1], '.',
                c=f"C{colids[i + 2]}")
        ax.plot(lpf.timea[sl] - tc + poffset[i] * p, fm[sl] + yoffsets[i + 1], 'k')
        if fmem is not None:
            ax.fill_between(lpf.timea[sl] - tc + poffset[i] * p,
                            fmem[sl] + yoffsets[i + 1],
                            fmep[sl] + yoffsets[i + 1],
                            fc=f"C{colids[i + 2]}", alpha=0.2)

    ax.text(-0.1, 1.0 + 5 * yoff, 'Spitzer 4.5 $\mu$m (2019)', ha='right')
    ax.text(0.42, 1.0 + 5 * yoff, 'Spitzer 4.5 $\mu$m (2014)', ha='right')
    ax.text(-0.1, 1.0 + 3 * yoff, 'Spitzer 3.6 $\mu$m (2019)', ha='right')
    ax.text(0.42, 1.0 + 3 * yoff, 'Spitzer 3.6 $\mu$m (2014)', ha='right')
    ax.text(0.42, 1.0 + 2 * yoff, 'CFHT Ks', ha='right')
    ax.text(0.42, 1.0 + 1 * yoff, 'LBT H', ha='right')
    ax.text(0.42, 0.9985, 'TESS', ha='right')

    setp(ax, xlim=(-0.75, 0.84), ylabel='Normalised flux')
    setp(ax, xlabel='Time - t$_0$ [d]')

    sb.despine(fig, offset=5)
    fig.tight_layout()
    return fig
