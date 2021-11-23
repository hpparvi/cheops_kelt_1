import pandas as pd
import seaborn as sb

from pathlib import Path

from matplotlib.pyplot import setp, subplots
from numpy import radians, argsort, floor

from pytransit import LinearModelBaseline, NormalPrior as NP
from pytransit.lpf.phasecurvelpf import PhaseCurveLPF
from pytransit.param.prior import LaplacePrior
from pytransit.utils.downsample import downsample_time_1d
from pytransit.orbits import epoch, fold

from src.kelt1 import read_tw_lightcurve, read_mcmc, beaming_amplitudes as ba, beaming_uncertainties as be

class CHEOPSLPF(PhaseCurveLPF):
    def __init__(self, scenario: str, savedir: Path = Path('results')):
        self.scenarios = 'a b c d e'.split()
        self.labels = {'a': 'emission_reflection_and_constrained_ev',
                       'b': 'emission_and_constrained_ev',
                       'c': 'emission_without_ev',
                       'd': 'emission_unconstrained_ev',
                       'e': 'emission_theoretical_ev'}

        if scenario not in self.scenarios:
            raise ValueError(f'The JointLPF scenario has to be one of {self.scenarios}')
        self.scenario = scenario

        time, flux, cov = [], [], []
        for v in range(1, 9):
            t, f, c = read_tw_lightcurve(v)
            time.extend(t)
            flux.extend(f)
            cov.extend(c)

        name = f"02{self.scenario}_cheops_{self.labels[self.scenario]}"
        super().__init__(name, 'cheops', time, flux, covariates=cov, tref=floor(time[0][0]), result_dir=savedir)
        self._add_baseline_model(LinearModelBaseline(self))

        for p in self.ps[self._sl_lm]:
            p.prior = LaplacePrior(p.prior.mean, p.prior.std)

        df = read_mcmc(self.result_dir / '01b_ext_emission_and_constrained_ev.nc')
        self.set_prior('tc', 'NP', df.tc.median(), df.tc.std())
        self.set_prior('p', 'NP', df.p.median(), df.p.std())
        self.set_prior('rho', 'NP', df.rho.median(), df.rho.std())
        self.set_prior('b', 'NP', df.b.median(), df.b.std())
        self.set_prior('k2', 'NP', df.k2.median(), df.k2.std())
        self.set_prior('oev', 'NP', df.oev.median(), df.oev.std())
        self.set_prior('secw', 'NP', 0.0, 1e-6)
        self.set_prior('sesw', 'NP', 0.0, 1e-6)
        self.set_prior('q1_cheops', 'NP', 0.5, 1e-6)
        self.set_prior('q2_cheops', 'NP', 0.5, 1e-6)

        self.set_prior('teo_cheops', 'NP', 0.0, 1e-7)
        self.set_prior('ten_cheops', 'NP', 1e-5, 1e-7)

        if self.scenario == 'a':
            self.set_prior('aev_cheops', 'NP', 1.128*df.aev_TESS.median(), 1.128*df.aev_TESS.std())
            self.set_prior('adb_cheops', 'NP', ba['CHEOPS'], 2*be['CHEOPS'])
            self.set_prior('ted_cheops', 'UP', 0.0, 0.2)
            self.set_prior('ag_cheops', 'UP', 0.0, 2.0)
        elif self.scenario == 'b':
            self.set_prior('aev_cheops', 'NP', 1.128*df.aev_TESS.median(), 1.128*df.aev_TESS.std())
            self.set_prior('adb_cheops', 'NP', ba['CHEOPS'], 2*be['CHEOPS'])
            self.set_prior('ted_cheops', 'UP', 0.0, 0.2)
            self.set_prior('ag_cheops', 'NP', 1e-4, 1e-6)
        elif self.scenario == 'c':
            self.set_prior('aev_cheops', 'NP', 1e-7, 1e-9)
            self.set_prior('adb_cheops', 'NP', 1e-7, 1e-9)
            self.set_prior('ted_cheops', 'UP', 0.0, 0.2)
            self.set_prior('ag_cheops', 'NP', 1e-4, 1e-6)
        elif self.scenario == 'd':
            self.set_prior('aev_cheops', 'NP', 1.128*df.aev_TESS.median(), 100e-6)
            self.set_prior('adb_cheops', 'NP', ba['CHEOPS'], 2*be['CHEOPS'])
            self.set_prior('ted_cheops', 'UP', 0.0, 0.2)
            self.set_prior('ag_cheops', 'NP', 1e-4, 1e-6)
        elif self.scenario == 'e':
            self.set_prior('aev_cheops', 'NP', 490e-6, 29e-6)
            self.set_prior('adb_cheops', 'NP', ba['CHEOPS'], 2*be['CHEOPS'])
            self.set_prior('ted_cheops', 'UP', 0.0, 0.2)
            self.set_prior('ag_cheops', 'NP', 1e-4, 1e-6)


    def plot_folded_light_curve(self, figsize=None):
        pv = self.de.minimum_location
        phase = fold(self.timea, pv[1], pv[0], 0.5)
        sids = argsort(phase)
        fm = self.transit_model(pv)
        bl = self.baseline(pv)
        bp, bf, be = downsample_time_1d(phase[sids], (self.ofluxa / bl)[sids], 10 / 60 / 24)
        fig, ax = subplots(figsize=figsize)
        ax.plot(phase, self.ofluxa / bl, '.', alpha=0.25)
        ax.plot(phase[sids], fm[sids], 'k')
        ax.errorbar(bp, bf, be, fmt='ok')
        fig.tight_layout()
        return fig