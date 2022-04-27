from pathlib import Path

import pandas as pd
import seaborn as sb
from george.kernels import ExpSquaredKernel as ESK
from matplotlib.pyplot import setp, subplots
from numpy import radians, argsort, percentile, median, array, concatenate, squeeze, log
from numpy.random import permutation
from pytransit import LinearModelBaseline, NormalPrior as NP
from pytransit.lpf.loglikelihood import CeleriteLogLikelihood
from pytransit.lpf.phasecurvelpf import PhaseCurveLPF
from pytransit.orbits import epoch, fold
from pytransit.utils.downsample import downsample_time_1d

from src.dataimport import load_tess, load_beatty_2017, load_spitzer, load_croll_2015
from src.georgelnl import GeorgeLogLikelihood
from src.kelt1 import (zero_epoch, period, beaming_amplitudes as ba, beaming_uncertainties as be,
                       ev_amplitudes as eva, star_rho, )

ds = dict(lbt=2., croll=2., spitzer=2., tess=None)

class ExternalDataLPF(PhaseCurveLPF):
    def __init__(self, scenario: str, savedir: Path = Path('results'), downsampling = None):
        """KELT-1b joint model LPF.

        This LPF models the TESS, LBT, and Spitzer data jointly.

        Scenarios:
          a) Unconstrained emission, EV amplitude constrained by a theoretical prior.
          b) Unconstrained emission, EV amplitude constrained by a ratio prior.
          c) Unconstrained emission and EV amplitude.
        """
        self.scenarios = 'a b c'.split()
        self.labels = {'a': 'emission_and_theoretical_ev',
                       'b': 'emission_and_constrained_ev',
                       'c': 'emission_and_unconstrained_ev'}

        self._sl_gp_Spitzer = None
        self._sl_gp_Ks = None
        self._sl_gp_H = None

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
        st, sf, scov, spb, snids = load_spitzer(downsampling=ds['spitzer'], nleg=1)

        # Store the H and Spitzer covariates for the GP model
        # ---------------------------------------------------
        self._gp_covs_h = hcov
        self._gp_covs_ks = kcov
        self._gp_covs_spitzer = scov
        hcov = [array([[]]) for i in range(len(hcov))]
        kcov = [array([[]]) for i in range(len(kcov))]
        scov = [array([[]]) for i in range(len(scov))]

        # Fix the noise IDs
        # -----------------
        hnids += 1+tnids[-1]
        knids += 1+hnids[-1]
        snids += 1+knids[-1]

        times = tt + ht + kt + st
        fluxes = tf + hf + kf + sf
        covs = tcov + hcov + kcov + scov
        pbs = pd.Categorical(tpb + hpb + kpb + spb, categories=['TESS', 'H', 'Ks', '36um', '45um'])
        noise_ids = concatenate([tnids, hnids, knids, snids])

        self.ins = array(pbs.categories.values)[pbs.codes]
        self.piis, c = [], 0
        ic = self.ins[0]
        for ins in self.ins:
            if ins == ic:
                c += 1
            else:
                ic = ins
                c = 0
            self.piis.append(c)

        # Initialise the PhaseCurveLPF and add in a linear baseline model
        # ---------------------------------------------------------------
        name = f"01{self.scenario}_ext_{self.labels[self.scenario]}"
        super().__init__(name, pbs.categories.values, times, fluxes, pbids=pbs.codes, covariates=covs,
                         wnids=noise_ids, result_dir=savedir)
        self._add_baseline_model(LinearModelBaseline(self))

    def _post_initialisation(self):
        super()._post_initialisation()
        self.set_prior('tc', 'NP', zero_epoch.n, 0.01)   # - Zero epoch: normal prior with an inflated uncertainty
        self.set_prior('p', 'NP', period.n, 3*period.s)  # - Orbital period: normal prior with an inflated uncertainty
        self.set_prior('rho', 'NP', star_rho.n, 3*star_rho.s)          # - Stellar density: wide normal prior
        self.set_prior('secw', 'NP', 0.0, 1e-6)          # - Circular orbit: sqrt(e) cos(w) and sqrt(e) sin(w) forced
        self.set_prior('sesw', 'NP', 0.0, 1e-6)          #   to zero with normal priors.
        self.set_prior('k2', 'NP', 0.078, 0.005)         # - Area ratio: wide normal prior based on Siverd et al. (2012)

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
        self.set_prior('oev', 'NP', 0.0, 0.09)
        for pb in self.passbands:
            self.set_prior(f'aev_{pb}', 'UP', 0.0, 1000e-6)      # - Ellipsoidal variation
            self.set_prior(f'log10_ted_{pb}', 'UP', -3.0, 0.0)   # - Emission dayside flux ratio
            self.set_prior(f'log10_ten_{pb}', 'UP', -4.0, 0.0)   # - Emission nightside flux ratio
            self.set_prior(f'teo_{pb}', 'NP', 0.0, radians(10))  # - Emission peak offset

        # Set the GP hyperparameter priors
        # --------------------------------
        def set_gp_hp_priors(gp, sl):
            for i, p in enumerate(self.ps[sl][:-1]):
                if i == 0:
                    log_output_scale = round(log(median([f.var() for f in gp.fluxes])), 2)
                    self.set_prior(p.name, 'NP', log_output_scale, 1.5)
                else:
                    self.set_prior(p.name, 'NP', 1.0, 1.0)

        self.set_prior('gp_TESS_ln_out', 'NP', round(log(self.fluxes[0].std()), 1), 1.0)
        self.set_prior('gp_TESS_ln_in', 'NP', 1.0, 1.0)

        set_gp_hp_priors(self._gp_h, self._sl_gp_H)
        set_gp_hp_priors(self._gp_ks, self._sl_gp_Ks)
        set_gp_hp_priors(self._gp_spitzer, self._sl_gp_Spitzer)

        # Set priors on the Doppler beaming amplitudes
        # --------------------------------------------
        # The Doppler beaming amplitudes are estimated from
        # BT-Settl model spectra for KELT-1 (see Appendix 2
        # notebook), and vary from 65 ppm in the CHEOPS passband
        # to 15 ppm in the Spitzer passbands.
        self.set_prior('adb_TESS', 'NP', ba['TESS'],   2*be['TESS'])
        self.set_prior('adb_H',    'NP', ba['H'],      2*be['H'])
        self.set_prior('adb_Ks',   'NP', ba['Ks'],     2*be['Ks'])
        self.set_prior('adb_36um', 'NP', ba['36um'],   2*be['36um'])
        self.set_prior('adb_45um', 'NP', ba['45um'],   2*be['45um'])

        # Set a prior on the geometric albedo
        # -----------------------------------
        for pb in self.passbands:
            self.set_prior(f'ag_{pb}', 'NP', 1e-4, 1e-6)

        # Set a prior on ellipsoidal variation
        # ------------------------------------
        # This prior dictates the ratio between
        # the ev amplitude in passband x and the
        # reference (TESS) passband. This ratio can
        # be estimated theoretically, as is done in
        # the "xxx" notebook.
        if self.scenario == 'a':
            for pb in self.passbands:
                self.set_prior(f'aev_{pb}', 'NP', eva[pb].n, eva[pb].s)
        elif self.scenario == "b":
            pr_e_h = NP(0.81, 0.01)
            pr_e_ks = NP(0.80, 0.01)
            pr_e_s1 = NP(0.77, 0.02)
            pr_e_s2 = NP(0.77, 0.02)
            def ev_amplitude_prior(pvp):
                h_ratio = pvp[:, 14] / pvp[:, 8]
                ks_ratio = pvp[:, 20] / pvp[:, 8]
                s1_ratio = pvp[:, 26] / pvp[:, 8]
                s2_ratio = pvp[:, 32] / pvp[:, 8]
                return pr_e_h.logpdf(h_ratio) + pr_e_ks.logpdf(ks_ratio) + pr_e_s1.logpdf(s1_ratio) + pr_e_s2.logpdf(s2_ratio)
            self.add_prior(ev_amplitude_prior)
        elif self.scenario == 'c':
            pass

        # Set a prior on the CHEOPS, H, and Ks nightside emission
        # -------------------------------------------------------
        # Nightside emission does not affect the results from the
        # passbands where we have only eclipse observations, so we
        # can safely fix the nightside amplitudes close to zero.
        self.set_prior('log10_ten_H', 'NP', -5, 0.01)
        self.set_prior('log10_ten_Ks', 'NP', -5, 0.01)

        # Force hot spot offsets close to similar values
        # ----------------------------------------------
        pr_emission_offset_difference = NP(0.0, 0.07)
        def emission_offset(pvp):
            eo_mean = pvp[:, [12, 18, 24, 30, 36]].mean()
            return (  pr_emission_offset_difference.logpdf(pvp[:, 12] - eo_mean)
                    + pr_emission_offset_difference.logpdf(pvp[:, 18] - eo_mean)
                    + pr_emission_offset_difference.logpdf(pvp[:, 24] - eo_mean)
                    + pr_emission_offset_difference.logpdf(pvp[:, 30] - eo_mean)
                    + pr_emission_offset_difference.logpdf(pvp[:, 36] - eo_mean))
        self.add_prior(emission_offset)

    # Define the log likelihood
    # -------------------------
    # We use a GP log likelihood for the TESS, LBT, and CFHT data and
    # uncorrelated normal log likelihood for Spitzer else.
    def _init_lnlikelihood(self):
        # TESS
        # ----
        self._add_lnlikelihood_model(CeleriteLogLikelihood(self, noise_ids=[0], name='gp_TESS'))

        # LBT H
        # -----
        c = self._gp_covs_h
        k = self.fluxes[1].var() * ESK(metric=1, ndim=c[0].shape[1], axes=0)
        for i in range(1, c[0].shape[1]):
            k *= ESK(metric=2, ndim=c[0].shape[1], axes=i)
        self._gp_h = GeorgeLogLikelihood(self, k, c, lcids=[1], name='gp_H')
        self._add_lnlikelihood_model(self._gp_h)

        # CFHT Ks
        k = self.fluxes[10].var() * ESK(metric=1, ndim=2, axes=0) * ESK(metric=1, ndim=2, axes=1)
        self._gp_ks = GeorgeLogLikelihood(self, k, self._gp_covs_ks, lcids=[2], name='gp_Ks')
        self._add_lnlikelihood_model(self._gp_ks)

        # Spitzer
        # -------
        k = median([f.var() for f in self.fluxes[3:]]) * ESK(metric=1, ndim=2, axes=0) * ESK(metric=1, ndim=2, axes=1)
        self._gp_spitzer = GeorgeLogLikelihood(self, k, self._gp_covs_spitzer, lcids=[3,4,5,6,7,8,9,10], name='gp_Spitzer')
        self._add_lnlikelihood_model(self._gp_spitzer)

    def lnposterior(self, pv):
        return squeeze(super().lnposterior(pv))
