import lmfit as lm
import scipy.constants as sc
import scipy.special as spec
from mkidcalculator.models.utils import scaled_alpha_inv
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

# special functions
log = np.log
real = np.real
imag = np.imag
digamma = spec.digamma


class Fr:
    """Basic fr model."""
    @classmethod
    def mattis_bardeen(cls, params, temperatures):
        """
        Returns the fractional frequency shift of the resonator with the
        specified model parameters for Mattis Bardeen effects.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            temperatures: numpy.ndarray
                The temperatures at which the fractional frequency shift is
                evaluated. If None, a fractional shift of zero is returned.
        Returns:
            dx: numpy.ndarray
                The fractional frequency shift.
        """
        if not HAS_SUPERCONDUCTIVITY:
            raise ImportError("The superconductivity package is not installed.")
        if temperatures is None:
            return 0.
        # unpack parameters
        alpha = params['alpha'].value
        tc = params['tc'].value
        bcs = params['bcs'].value
        gamma = params['gamma'].value
        limit = params['limit'].value
        # calculate dx
        sigma0 = cc.value(0, f0, tc, gamma=gamma, bcs=bcs, low_energy=False)
        sigma1 = cc.value(temperatures, f0, tc, gamma=gamma, bcs=bcs, low_energy=False)
        dx = -0.5 * alpha * limit * imag((sigma1 - sigma0) / sigma0)  # take imaginary part after in case gamma != 0
        return dx

    @classmethod
    def two_level_systems(cls, params, temperatures, powers):
        """
        Returns the fractional frequency shift of the resonator with the
        specified model parameters for two-level system effects.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            temperatures: numpy.ndarray
                The temperatures at which the fractional frequency shift is
                evaluated. If None, only the power dependence is used.
            powers: numpy.ndarray
                The powers at which the fractional frequency shift is
                evaluated. If None, only the temperature dependence is used.
        Returns:
            dx: numpy.ndarray
                The fractional frequency shift.
        """
        # unpack parameters
        f0 = params['f0'].value
        fd = params['fd'].value
        pc = params['pc'].value
        # calculate dx
        dx = 0
        if temperatures is not None:
            xi = h * f0 / (2 * kb * temperatures)
            dx += fd / pi * (real(digamma(0.5 + xi / (1j * pi))) - log(2 * xi))
        if powers is not None:
            dx /= sqrt(1. + 10 ** ((powers - pc) / 10.))
        return dx

    @classmethod
    def constant_offset(cls, params):
        """
        Returns the constant offset frequency for zero frequency shifts.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
        Returns:
            f0: float
                The offset frequency.
        """
        return params['f0'].value

    @classmethod
    def model(cls, params, temperatures=None, powers=None):
        """
        Returns the model of fr for the specified model parameters.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            temperatures: numpy.ndarray (optional)
                The temperatures at which the fractional frequency shifts are
                evaluated. The default is None and fractional shifts due to
                temperature are ignored.
            powers: numpy.ndarray (optional)
                The powers at which the fractional frequency shifts are
                evaluated. The default is None and fractional shifts due to
                power are ignored.
        Returns:
            fr: numpy.ndarray
                The resonance frequency.
        """
        f0 = cls.constant_offset(params)
        dx = cls.mattis_bardeen(params, temperatures)
        dx += cls.two_level_systems(params, temperatures, powers)
        fr = f0 * dx + f0
        return fr

    @classmethod
    def residual(cls, params, data, temperatures=None, powers=None, sigmas=None):
        """
        Return the normalized residual between the fr data and model.
        Args:
            params: lmfit.Parameters() object
                The parameters for the model function.
            data: numpy.ndarray
                The fr data.
            temperatures: numpy.ndarray (optional)
                The temperatures at which the fractional frequency shifts are
                evaluated. The default is None and fractional shifts due to
                temperature are ignored.
            powers: numpy.ndarray (optional)
                The powers at which the fractional frequency shifts are
                evaluated. The default is None and fractional shifts due to
                power are ignored.
            sigmas: numpy.ndarray (optional)
                The error associated with each data point. The default is None
                and the residual is not normalized.
        Returns:
            residual: numpy.ndarray
                the normalized residuals.
        """
        fr = cls.model(params, temperatures=temperatures, powers=powers)
        residual = (fr - data) / sigmas if sigmas is not None else (fr - data)
        return residual

    @classmethod
    def guess(cls, data, tc, alpha=0.5, bcs=BCS, fd=1e-5, powers=None, limit=-1, fit_resonance=True, fit_mbd=True,
              fix_tc=True, fit_alpha=False, fit_dynes=False, fit_tls=True, fit_fd=True, fit_pc=True):
        """
        Guess the model parameters based on the data. Returns a
        lmfit.Parameters() object.
        Args:
            data: numpy.ndarray
                The fr data.
            tc: float
                The transition temperature for the resonator in Kelvin.
            alpha: float (optional)
                The kinetic inductance fraction. The default is 0.5.
            bcs: float (optional)
                ∆ = bcs * kB * tc. The default is the usual BCS value.
            fd: float (optional)
                The TLS fraction and loss tangent factor. The default is
                1e-5.
            powers: numpy.ndarray (optional)
                The powers at which the fr data is taken. The default is None.
                If specified, this helps set the pc parameter to a reasonable
                value.
            limit: float (optional)
                The float corresponding to the superconducting limit of the
                resonator. The default is -1 which corresponds to the thin
                film, local limit. -1/2 is the thick film, local limit. -1/3 is
                the thick film, extreme anomalous limit.
            fit_resonance: boolean (optional)
                A boolean specifying if the offset frequency should be varied
                during the fit. The default is True.
            fit_mbd: boolean (optional)
                A boolean specifying whether the Mattis-Bardeen parameters
                should be varied during the fit. The default is True.
            fix_tc: boolean (optional)
                A boolean specifying whether to fix Tc in the fit. bcs is
                varied instead. The default is True. Tc and bcs may still be
                fixed if fit_mbd is False.
            fit_alpha: boolean (optional)
                A boolean specifying whether to vary alpha during the fit. The
                default is True. alpha may still be not fit if fit_mbd is
                False.
            fit_dynes: boolean (optional)
                A boolean specifying whether to vary the dynes parameter in the
                fit. The default is False.
            fit_tls: boolean (optional)
                A boolean specifying whether to vary the TLS parameters during
                the fit. The default is True.
            fit_fd: boolean (optional)
                A boolean specifying whether to vary the TLS fraction and loss
                tangent factor during the fit. The default is True. fd may
                still not be varied if fit_tls is False.
            fit_pc: boolean (optional)
                A boolean specifying whether to vary the critical power during
                the fit. The default is True. pc may still not be varied if
                fit_tls is False.
        Returns:
            params: lmfit.Parameters
                An object with guesses and bounds for each parameter.
        """
        pc = np.mean(powers) if powers is not None else 0
        gamma_sqrt = 0.1 if fit_dynes is True else fit_dynes
        f0 = np.max(data)

        # make the parameters object (coerce all values to float to avoid ints and numpy types)
        params = lm.Parameters(usersyms={'scaled_alpha_inv': scaled_alpha_inv})
        # resonator params
        params.add("f0", value=float(f0), vary=fit_resonance)
        params.add("limit", value=float(limit), vary=False)
        # Mattis-Bardeen params
        params.add("tc", value=float(tc), vary=fit_mbd and not fix_tc)
        params.add("bcs", value=float(bcs), vary=fit_mbd and fix_tc)
        params.add("scaled_alpha", value=float(scaled_alpha(alpha)), vary=fit_mbd and fit_alpha)
        # Dynes params
        params.add("gamma_sqrt", value=float(gamma_sqrt), vary=bool(fit_dynes))
        # two level system params
        params.add("fd_sqrt", value=float(sqrt(fd) if fit_tls else 0.), vary=fit_tls and fit_fd)
        params.add("pc", value=float(pc), vary=fit_tls and fit_pc)
        # derived params
        params.add("alpha", expr='scaled_alpha_inv(scaled_alpha)')
        params.add("fd", expr='fd_sqrt**2')
        params.add("gamma", expr="gamma_sqrt**2")
        return params
