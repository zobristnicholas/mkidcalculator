import numpy as np
import lmfit as lm
import scipy.constants as sc
from mkidcalculator.models.utils import scaled_alpha, scaled_alpha_inv

try:
    from superconductivity import complex_conductivity as cc
    HAS_SUPERCONDUCTIVITY = True
except ImportError:
    HAS_SUPERCONDUCTIVITY = False

# constants
pi = sc.pi
kb = sc.k  # [J / K] Boltzmann const
h = sc.h  # [J * s] Plank constant
BCS = pi / np.exp(np.euler_gamma)  # bcs constant


class Qi:
    @classmethod
    def mattis_bardeen(cls, params, temperatures, low_energy=False, parallel=False):
        if not HAS_SUPERCONDUCTIVITY:
            raise ImportError("The superconductivity package is not installed.")
        if temperatures is None:
            return 0.
        # unpack parameters
        alpha = params['alpha'].value
        tc = params['tc'].value
        bcs = params['bcs'].value
        dynes = params['dynes'].value
        gamma = np.abs(params['gamma'].value)
        f0 = params['f0'].value
        # calculate Qinv
        sigma0 = cc.value(0, f0, tc, gamma=dynes, bcs=bcs, low_energy=low_energy, parallel=parallel)
        sigma1 = cc.value(temperatures, f0, tc, gamma=dynes, bcs=bcs, low_energy=low_energy, parallel=parallel)
        q_inv = -alpha * gamma * np.real((sigma1 - sigma0) / sigma0**(gamma + 1)) / np.imag(sigma0**-gamma)
        q_inv += alpha * np.real(sigma0**-gamma) / np.imag(sigma0**-gamma)  # Qinv(0)
        return q_inv

    @classmethod
    def two_level_systems(cls, params, temperatures, powers):
        if powers is None and temperatures is None:
            return 0.
        # unpack parameters
        f0 = params['f0'].value
        fd = params['fd'].value
        pc = params['pc'].value
        # calculate Qinv
        q_inv = fd
        if temperatures is not None:
            xi = h * f0 / (2 * kb * temperatures)
            q_inv *= np.tanh(xi)
        if powers is not None:
            q_inv /= np.sqrt(1. + 10**((powers - pc) / 10.))
        return q_inv

    @classmethod
    def constant_loss(cls, params):
        return params['q0_inv'].value

    @classmethod
    def model(cls, params, temperatures=None, powers=None, low_energy=False, parallel=False):
        q_inv = cls.mattis_bardeen(params, temperatures, low_energy=low_energy, parallel=parallel)
        q_inv += cls.two_level_systems(params, temperatures, powers)
        q_inv += cls.constant_loss(params)
        return 1 / q_inv

    @classmethod
    def residual(cls, params, data, temperatures=None, powers=None, sigmas=None, low_energy=False, parallel=False):
        q = cls.model(params, temperatures=temperatures, powers=powers, low_energy=low_energy, parallel=parallel)
        residual = (q - data) / sigmas if sigmas is not None else (q - data)
        return residual

    @classmethod
    def guess(cls, data, f0, tc, alpha=0.5, bcs=BCS, temperatures=None, powers=None, gamma=1., fit_resonance=False,
              fit_mb=True, fit_tc=False, fit_alpha=True, fit_dynes=False, fit_tls=True, fit_fd=True, fit_pc=False,
              fit_loss=False):
        scale = 2 if fit_tls and fit_loss else 1
        # guess constant loss
        qi_inv_min = np.min(1 / data)
        q0_inv = qi_inv_min / scale if fit_loss else 0
        # guess two level system values
        if temperatures is not None:
            xi = h * f0 / (2 * kb * temperatures)
            fd = np.min(qi_inv_min / np.tanh(xi)) / scale if fit_tls else 0
        else:
            fd = 0
        pc = np.mean(powers) if powers is not None else 0
        dynes_sqrt = np.sqrt(0.01) if fit_dynes is True else np.sqrt(fit_dynes)

        # make the parameters object (coerce all values to float to avoid ints and numpy types)
        params = lm.Parameters(usersyms={'scaled_alpha_inv': scaled_alpha_inv})
        # resonator params
        params.add("f0", value=float(f0), vary=fit_resonance, min=0)
        params.add("gamma", value=float(gamma), vary=False)
        # constant loss parameters
        params.add("q0_inv", value=float(q0_inv), vary=fit_loss)
        # two level system params
        params.add("fd_scaled", value=float(np.sqrt(fd * 1e6)), vary=fit_tls and fit_fd)
        params.add("pc", value=float(pc) if fit_pc else np.inf, vary=fit_tls and fit_pc)
        # Mattis-Bardeen params
        params.add("tc", value=float(tc), vary=fit_mb and fit_tc, min=0)
        params.add("bcs", value=float(bcs), vary=fit_mb and not fit_tc, min=0)
        params.add("scaled_alpha", value=float(scaled_alpha(alpha)), vary=fit_mb and fit_alpha)
        # Dynes params
        params.add("dynes_sqrt", value=float(dynes_sqrt), vary=bool(fit_dynes))
        # derived params
        params.add("alpha", expr='scaled_alpha_inv(scaled_alpha)')
        params.add("fd", expr='fd_scaled**2 * 1e-6')
        params.add("dynes", expr="dynes_sqrt**2")

        return params
