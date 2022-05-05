from astropy.stats import mad_std
from numpy import zeros_like, squeeze, zeros, isfinite, inf, sqrt, diff, median, log10

try:
    from george import GP

    with_george = True
except ImportError:
    with_george = False

from pytransit.param import LParameter, NormalPrior as NP, UniformPrior as UP


class GeorgeLogLikelihood:
    def __init__(self, lpf, kernel, covs, lcids, name: str = 'gp'):
        if not with_george:
            raise ImportError("GeorgeLogLikelihood requires George.")
        if lpf.lcids is None:
            raise ValueError('The LPF data needs to be initialised before initialising CeleriteLogLikelihood.')

        self.lpf = lpf
        self.name = name
        self.kernel = kernel
        self.nhp = len(kernel.get_parameter_names())
        self.lcids = lcids

        self.covs = covs
        self.slices = []
        self.fluxes = []
        self.times = []

        for lcid in self.lcids:
            sl = lpf.lcslices[lcid]
            self.slices.append(sl)
            self.times.append(lpf.timea[sl])
            self.fluxes.append(lpf.ofluxa[sl])

        self.gp = GP(kernel)
        self.init_parameters()

    def init_parameters(self):
        name = self.name
        wns = log10(median([mad_std(diff(f)) / sqrt(2) for f in self.fluxes]))
        pgp = [LParameter(f'{name}_p_{i}', f'{name} hyperparameter {i}', '', NP(-6, 1.5), bounds=(-inf, inf)) for i in
               range(self.nhp)]
        pgp.append(
            LParameter(f'{name}_log10_wn', f'{name} log10 white noise sigma', '', NP(wns, 0.025), bounds=(-inf, inf)))
        self.lpf.ps.thaw()
        self.lpf.ps.add_global_block(self.name, pgp)
        self.lpf.ps.freeze()
        self.pv_slice = self.lpf.ps.blocks[-1].slice
        self.pv_start = self.lpf.ps.blocks[-1].start
        setattr(self.lpf, f"_sl_{name}", self.pv_slice)
        setattr(self.lpf, f"_start_{name}", self.pv_start)

    def lnlikelihood(self, pv, model):
        parameters = pv[self.pv_slice]
        self.gp.set_parameter_vector(parameters[:-1])
        yerr = 10 ** parameters[-1]
        lnl = 0.0
        try:
            for fobs, sl, covs in zip(self.fluxes, self.slices, self.covs):
                self.gp.compute(covs, yerr=yerr)
                lnl += self.gp.log_likelihood(fobs - model[sl])
            return lnl
        except (ValueError, ZeroDivisionError):
            return -inf

    def predict_baseline(self, pv):
        parameters = pv[self.pv_slice]
        self.gp.set_parameter_vector(parameters[:-1])
        yerr = 10 ** parameters[-1]
        bl = zeros_like(self.lpf.timea)
        for fobs, sl, covs in zip(self.fluxes, self.slices, self.covs):
            self.gp.compute(covs, yerr=yerr)
            residuals = fobs - squeeze(self.lpf.flux_model(pv))[sl]
            bl[sl] = self.gp.predict(residuals, covs, return_cov=False)
        return 1. + bl

    def __call__(self, pvp, model):
        if pvp.ndim == 1:
            lnlike = self.lnlikelihood(pvp, model)
        else:
            lnlike = zeros(pvp.shape[0])
            for ipv, pv in enumerate(pvp):
                if all(isfinite(model[ipv])):
                    lnlike[ipv] = self.lnlikelihood(pv, model[ipv])
                else:
                    lnlike[ipv] = -inf
        return lnlike