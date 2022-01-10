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

class FinalLPF(PhaseCurveLPF):
    def __init__(self, scenario: str, savedir: Path = Path('results'), downsampling = None):
        """KELT-1b joint model LPF.

        This LPF models the CHEOPS, TESS, LBT, CFHT, and Spitzer data jointly.

        Scenarios:
          a) Zero albedo and unconstrained emission, EV amplitude constrained by a ratio prior.
        """
        self.scenarios = ('a', 'b')
        self.labels = {'a': 'emission_and_constrained_ev',
                       'b': 'emission_and_reflection_and_constrained_ev'}

        if scenario not in self.scenarios:
            raise ValueError(f'The FinalLPF scenario has to be one of {self.scenarios}')
        self.scenario = scenario

        if downsampling is not None:
            ds.update(downsampling)

        # Load in the data
        # ----------------
        ct, cf, ccov, cpb, cnids = load_detrended_cheops()
        tt, tf, tcov, tpb, tnids = load_tess(downsampling=ds['tess'])
        ht, hf, hcov, hpb, hnids = load_beatty_2017(downsampling=ds['lbt'])
        kt, kf, kcov, kpb, knids = load_croll_2015(downsampling=ds['croll'])
        st, sf, scov, spb, snids = load_spitzer(downsampling=ds['spitzer'], nleg=3)

        # Fix the noise IDs
        # -----------------
        hnids += 1+tnids[-1]
        knids += 1+hnids[-1]
        snids += 1+knids[-1]

        times = ct + tt + ht + kt + st
        fluxes = cf + tf + hf + kf + sf
        covs = ccov + tcov + hcov + kcov + scov
        pbs = pd.Categorical(cpb + tpb + hpb + kpb + spb, categories=['CHEOPS', 'TESS', 'H', 'Ks', '36um', '45um'])
        noise_ids = concatenate([cnids, tnids, hnids, knids, snids])

        # Initialise the PhaseCurveLPF and add in a linear baseline model
        # ---------------------------------------------------------------
        name = f"03{self.scenario}_fin_{self.labels[self.scenario]}"
        super().__init__(name, pbs.categories.values, times, fluxes, pbids=pbs.codes, covariates=covs,
                         wnids=noise_ids, result_dir=savedir)
        self._add_baseline_model(LinearModelBaseline(self))

    def _post_initialisation(self):
        super()._post_initialisation()
        self.set_prior('tc', 'NP', zero_epoch.n, 0.01)   # - Zero epoch: normal prior with an inflated uncertainty
        self.set_prior('p', 'NP', period.n, 3*period.s)  # - Orbital period: normal prior with an inflated uncertainty
        self.set_prior('rho', 'NP', 0.6, 0.075)          # - Stellar density: wide normal prior
        self.set_prior('secw', 'NP', 0.0, 1e-6)          # - Circular orbit: sqrt(e) cos(w) and sqrt(e) sin(w) forced
        self.set_prior('sesw', 'NP', 0.0, 1e-6)          #   to zero with normal priors.
        self.set_prior('k2', 'NP', 0.078, 0.005)         # - Area ratio: wide normal prior based on Siverd et al. (2012)

        # Set the limb darkening priors
        # -----------------------------
        # The priors for quadratic limb darkening coefficients
        # have been calculated using LDTk.
        self.set_prior('q1_36um', 'NP', 0.03, 0.0005)
        self.set_prior('q2_36um', 'NP', 0.28, 0.0040)
        self.set_prior('q1_45um', 'NP', 0.02, 0.0004)
        self.set_prior('q2_45um', 'NP', 0.29, 0.0050)

        self.set_prior('q1_CHEOPS',  'NP', 0.5, 0.001)
        self.set_prior('q2_CHEOPS',  'NP', 0.5, 0.001)
        self.set_prior('q1_H',  'NP', 0.5, 0.001)
        self.set_prior('q2_H',  'NP', 0.5, 0.001)
        self.set_prior('q1_Ks', 'NP', 0.5, 0.001)
        self.set_prior('q2_Ks', 'NP', 0.5, 0.001)

        # Set the default phase curve priors
        # ----------------------------------
        self.set_prior('oev', 'NP', 0.0, 1e-7)
        for pb in self.passbands:
            self.set_prior(f'aev_{pb}', 'UP', 0.0, 1000e-6)      # - Ellipsoidal variation
            self.set_prior(f'ted_{pb}', 'UP', 0.0, 0.5)          # - Emission day-side flux ratio
            self.set_prior(f'ten_{pb}', 'UP', 0.0, 0.5)          # - Emission night-side flux ratio
            self.set_prior(f'teo_{pb}', 'NP', 0.0, radians(10))  # - Emission peak offset

        # Force the CHEOPS, H, and Ks band emission offset to zero
        # --------------------------------------------------------
        # We do this because we don't have enough phase
        # coverage to measure the offsets.
        self.set_prior('teo_CHEOPS', 'NP', 0.0, 1e-5)
        self.set_prior('teo_H', 'NP', 0.0, 1e-5)
        self.set_prior('teo_Ks', 'NP', 0.0, 1e-5)

        # Set priors on the Doppler beaming amplitudes
        # --------------------------------------------
        # The Doppler beaming amplitudes are estimated from
        # BT-Settl model spectra for KELT-1 (see Appendix 2
        # notebook), and vary from 65 ppm in the CHEOPS passband
        # to 15 ppm in the Spitzer passbands.
        self.set_prior('adb_CHEOPS', 'NP', ba['CHEOPS'], 2*be['CHEOPS'])
        self.set_prior('adb_TESS',   'NP', ba['TESS'], 2*be['TESS'])
        self.set_prior('adb_H',      'NP', ba['H'],    2*be['H'])
        self.set_prior('adb_Ks',     'NP', ba['Ks'],   2*be['Ks'])
        self.set_prior('adb_36um',   'NP', ba['36um'],   2*be['36um'])
        self.set_prior('adb_45um',   'NP', ba['45um'],   2*be['45um'])

        # Set a prior on the geometric albedo
        # -----------------------------------
        if self.scenario == 'a':
            for pb in self.passbands:
                self.set_prior(f'ag_{pb}', 'NP', 1e-4, 1e-6)
        elif self.scenario == 'b':
            for pb in self.passbands:
                self.set_prior(f'ag_{pb}', 'UP', 0.0, 2.0)

        # Set a prior on ellipsoidal variation
        # ------------------------------------
        # This prior dictates the ratio between
        # the ev amplitude in passband x and the
        # reference (TESS) passband. This ratio can
        # be estimated theoretically, as is done in
        # the "xxx" notebook.
        pr_e_cheops = NP(1.083, 0.01)
        pr_e_h = NP(0.81, 0.01)
        pr_e_ks = NP(0.80, 0.01)
        pr_e_s1 = NP(0.77, 0.01)
        pr_e_s2 = NP(0.77, 0.01)
        def ev_amplitude_prior(pvp):
            ch_ratio = pvp[: 14] / pvp[:, 8] # TODO: Fix me!
            h_ratio = pvp[:, 14] / pvp[:, 8]
            ks_ratio = pvp[:, 20] / pvp[:, 8]
            s1_ratio = pvp[:, 26] / pvp[:, 8]
            s2_ratio = pvp[:, 32] / pvp[:, 8]
            return (pr_e_cheops.logpdf(ch_ratio) + pr_e_h.logpdf(h_ratio) + pr_e_ks.logpdf(ks_ratio)
                    + pr_e_s1.logpdf(s1_ratio) + pr_e_s2.logpdf(s2_ratio))
        self.add_prior(ev_amplitude_prior)


    # Define the log likelihood
    # -------------------------
    # We use a GP log likelihood for the TESS, LBT, and CFHT data and iid normal
    # log likelihood for CHEOPS and Spitzer.
    def _init_lnlikelihood(self):
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[1]))  # TESS
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[2]))  # LBT H
            self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[3]))  # CFHT Ks
            self._add_lnlikelihood_model(WNLogLikelihood(self, noise_ids=array([0,4,5,6,7])))
