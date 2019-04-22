import logging
import numpy as np
import lmfit as lm
import scipy.signal as sps

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def baseline(params, f):
    """
    Return S21 at the frequencies f, ignoring the effect of the resonator, for
    the specified model parameters.
    Args:
        params: lmfit.Parameters() object
            The parameters for the model function.
        f: numpy.ndarray, dtype=real, shape=(N,)
            Frequency points corresponding to z.
    Returns:
        z: numpy.ndarray
            The S21 scattering parameter.
    """
    # 0th, 1st, and 2nd terms in a power series to handle magnitude gain different than 1
    gain0 = params['gain0'].value
    gain1 = params['gain1'].value
    gain2 = params['gain2'].value

    # 0th and 1st terms in a power series to handle phase gain different than 1
    phase0 = params['phase0'].value
    phase1 = params['phase1'].value
    phase2 = params['phase2'].value

    # the gain should be referenced to the file midpoint so that the baseline
    # coefficients do not drift with changes in f0
    # this breaks down if different sweeps have different frequency ranges
    fm = f[int(np.round((len(f) - 1) / 2.0))]
    ffm = (f - fm) / fm

    # Calculate magnitude and phase gain
    gain = gain0 + gain1 * ffm + gain2 * ffm**2
    phase = np.exp(1j * (phase0 + phase1 * ffm + phase2 * ffm**2))
    z = gain * phase
    return z


def resonance(params, f):
    """
    Return S21 at the frequencies f, only considering the resonator, for the
    specified model parameters.
    Args:
        params: lmfit.Parameters() object
            The parameters for the model function.
        f: numpy.ndarray, dtype=real, shape=(N,)
            Frequency points corresponding to z.
    Returns:
        z: numpy.ndarray
            The S21 scattering parameter.
    """
    df = params['df'].value  # frequency shift due to mismatched impedances
    f0 = params['f0'].value  # resonant frequency
    qc = params['qc'].value  # coupling Q
    qi = params['qi'].value  # internal Q
    a = params['a'].value  # nonlinearity parameter (Swenson et al. 2013)

    # calculate the total Q
    q0 = 1. / (1. / qi + 1. / qc)

    # Make everything referenced to the shifted, unitless, reduced frequency
    # accounting for nonlinearity
    if a != 0:
        y0 = q0 * (f - f0) / f0
        select_root = np.min if f[0] - f[1] > 0 else np.max  # min is low to high, max is high to low sweep
        y = np.zeros(y0.shape)
        for ii, y0_ii in enumerate(y0):
            roots = np.roots([4, -4 * y0_ii, 1, -(y0_ii + a)])
            y[ii] = select_root(roots[np.isreal(roots)].real)
        ff = y / q0
    else:
        ff = (f - f0) / f0

    z = (1. / qi + 1j * 2.0 * (ff + df / f0)) / (1. / q0 + 1j * 2.0 * ff)
    return z


def mixer(params, z):
    """
    Apply the mixer correction specified in the model parameters to S21.
    Args:
        params: lmfit.Parameters() object
            The parameters for the model function.
        z: numpy.ndarray, dtype=complex, shape=(N,)
            Complex resonator scattering parameter.
    Returns:
        z: numpy.ndarray
            The S21 scattering parameter.
    """
    alpha = params['alpha'].value
    gamma = params['gamma'].value
    offset = params['i_offset'].value + 1j * params['q_offset'].value
    z = (z.real * alpha + 1j * (z.real * np.sin(gamma) + z.imag * np.cos(gamma))) + offset
    return z


def mixer_inverse(params, z):
    """
    Undo the mixer correction specified by the parameters to S21. This is
    useful for removing the effect of the mixer on real data.
    Args:
        params: lmfit.Parameters() object or tuple
            The parameters for the model function or a tuple with
            (alpha, gamma, offset) where offset is i_offset + i * q_offset.
        z: numpy.ndarray, dtype=complex, shape=(N,)
            Complex resonator scattering parameter.
    Returns:
        z: numpy.ndarray
            The S21 scattering parameter.
    """
    if isinstance(params, lm.Parameters):
        alpha = params['alpha'].value
        gamma = params['gamma'].value
        offset = params['i_offset'].value + 1j * params['q_offset'].value
    else:
        alpha, gamma, offset = params
    z -= offset
    z = (z.real / alpha + 1j * (-z.real * np.tan(gamma) / alpha + z.imag / np.cos(gamma)))
    return z


def calibrate(params, z, f, mixer_correction=True):
    """
    Remove the baseline and mixer effects from the S21 data.
    Args:
        params: lmfit.Parameters() object
            The parameters for the model function.
        z: numpy.ndarray, dtype=complex, shape=(N,)
            Complex resonator scattering parameter.
        f: numpy.ndarray, dtype=real, shape=(N,)
            Frequency points corresponding to z.
        mixer_correction: bool (optional)
            Remove the mixer correction specified in the params object. The
            default is True.
    Returns:
        z: numpy.ndarray
            The S21 scattering parameter.
    """
    if mixer_correction:
        z = mixer_inverse(params, z) / baseline(params, f)
    else:
        z /= baseline(params, f)
    return z


def model(params, f, mixer_correction=True):
    """
    Return the model of S21 at the frequencies f for the specified model
    parameters.
    Args:
        params: lmfit.Parameters() object
            The parameters for the model function.
        f: numpy.ndarray, dtype=real, shape=(N,)
            Frequency points corresponding to z.
        mixer_correction: bool (optional)
            Apply the mixer correction specified in the params object. The
            default is True.
    Returns:
        z: numpy.ndarray
            The S21 scattering parameter.
    """
    z = baseline(params, f) * resonance(params, f)
    if mixer_correction:
        z = mixer(params, z)
    return z


def residual(params, z, f, sigma=None, return_real=True):
    """
    Return the normalized residual between the S21 data and model.
    Args:
        params: lmfit.Parameters() object
            The parameters for the model function.
        z: numpy.ndarray, dtype=complex, shape=(N,)
            Complex resonator scattering parameter.
        f: numpy.ndarray, dtype=real, shape=(N,)
            Frequency points corresponding to z.
        sigma: numpy.ndarray, dtype=complex, shape=(N,)
            The standard deviation of the data z at f in the form
            std(z.real) + i std(z.imag). The default is None. If None is
            provided, the standard deviation is calculated from the first 10
            points after being detrended.
        return_real: bool (optional)
            Concatenate the real and imaginary parts of the residual into a
            real 1D array of shape (2N,).
    Returns:
        res: numpy.ndarray, dtype=(complex or float)
            Either a complex N or a real 2N element 1D array (depending on
            return_real) with the normalized residuals.
    """
    # grab the model
    m = model(params, f)
    # calculate constant error from standard deviation of the first 10 pts of the data if not supplied
    if sigma is None:
        eps_real = np.std(sps.detrend(z.real[0:10]), ddof=1)
        eps_imag = np.std(sps.detrend(z.imag[0:10]), ddof=1)
        # make sure there are no zeros
        if eps_real == 0:
            log.warning("zero variance calculated and set to 1 when detrending I data")
            eps_real = 1
        if eps_imag == 0:
            log.warning("zero variance calculated and set to 1 when detrending Q data")
            eps_imag = 1
        sigma = np.full_like(z, eps_real + 1j * eps_imag)
    if return_real:
        # convert model, data, and error into a real vector
        m_1d = np.concatenate((m.real, m.imag), axis=0)
        z_1d = np.concatenate((z.real, z.imag), axis=0)
        sigma_1d = np.concatenate((sigma.real, sigma.imag), axis=0)
        res = (m_1d - z_1d) / sigma_1d
    else:
        # return the complex residual
        res = (m.real - z.real) / sigma.real + 1j * (m.imag - z.imag) / sigma.imag
    return res


def guess(z, f, mixer_imbalance=None, mixer_offset=0, use_filter=False, filter_length=None, fit_resonance=True,
          nonlinear_resonance=False, fit_gain=True, quadratic_gain=True, fit_phase=True, quadratic_phase=False,
          fit_imbalance=False, fit_offset=False):
    """
    Guess the model parameters based on the data. Returns a lmfit.Parameters()
    object.
    Args:
        z: numpy.ndarray, dtype=complex, shape=(N,)
            Complex resonator scattering parameter.
        f: numpy.ndarray, dtype=real, shape=(N,)
            Frequency points corresponding to z.
        mixer_imbalance: numpy.ndarray, dtype=complex, shape=(3, M) (optional)
            Mixer calibration data (three data sets of I and Q beating). The
            three sets correspond to the frequencies at the beginning middle
            and end frequencies. The default is None, which means no
            calibration is assumed.
        mixer_offset: complex, iterable (optional)
            A complex number corresponding to the I + iQ mixer offset. The
            default is 0, corresponding to no offset. If the input is iterable,
            a mean is taken to determine the mixer_offset value.
        use_filter: bool (optional)
            Filter the phase and magnitude data of z before trying to guess the
            parameters. This can be helpful for noisy data, but can also result
            in poor guesses for clean data. The default is False.
        filter_length: int, odd >= 3 (optional)
            If use_filter==True, this is used as the filter length. Only odd
            numbers greater or equal to three are allowed. If None, a
            filter length is computed as roughly 1% of the number of points
            in z. The default is None.
        fit_resonance: bool (optional)
            Allow the resonance parameters to vary in the fit. The default is
            True.
        nonlinear_resonance: bool (optional)
            Allow the resonance model to fit for nonlinear behavior. The
            default is True.
        fit_gain: bool (optional)
            Allow the gain parameters to vary in the fit. The default is True.
        quadratic_gain: bool (optional)
            Allow for a quadratic gain component in the model. The default is
            True.
        fit_phase: bool (optional)
            Allow the phase parameters to vary in the fit. The default is True.
        quadratic_phase: bool (optional)
            Allow for a quadratic phase component in the model. The default is
            False since there isn't an obvious physical reason why there should
            be a quadratic term.
        fit_imbalance: bool (optional)
            Allow the IQ mixer amplitude and phase imbalance to vary in the
            fit. The default is False. The imbalance is typically calibrated
            and not fit.
        fit_offset: bool (optional)
            Allow the IQ mixer offset to vary in the fit. The default is False.
            The offset is highly correlated with the gain parameters and
            typically should not be allowed to vary unless the gain is properly
            calibrated.
    Returns:
        params: lmfit.Parameters
            An object with guesses and bounds for each parameter.
    """
    # undo the mixer calibration for more accurate guess if known ahead of time
    mixer_offset = np.mean(mixer_offset)
    if mixer_imbalance is not None:
        i, q = mixer_imbalance.real, mixer_imbalance.imag
        # bandpass filter the I signal
        fft_i = np.fft.rfft(i)
        fft_i[:, 0] = 0
        indices = np.array([np.arange(fft_i[0, :].size)] * 3)
        f_i_ind = np.argmax(np.abs(fft_i), axis=-1)[:, np.newaxis]
        fft_i[np.logical_or(indices < f_i_ind - 1, indices > f_i_ind + 1)] = 0
        ip = np.fft.irfft(fft_i, i[0, :].size)
        # bandpass filter the Q signal
        fft_q = np.fft.rfft(q)
        fft_q[:, 0] = 0
        f_q_ind = np.argmax(np.abs(fft_q), axis=-1)[:, np.newaxis]
        fft_q[np.logical_or(indices < f_q_ind - 1, indices > f_q_ind + 1)] = 0
        qp = np.fft.irfft(fft_q, q[0, :].size)
        # TODO: clip Ip and Qp so that they are approximately N periods long to minimize errors
        # compute alpha and gamma
        amp = np.sqrt(2 * np.mean(qp**2, axis=-1))
        alpha = np.sqrt(2 * np.mean(ip**2, axis=-1)) / amp
        ratio = np.angle(np.fft.rfft(ip)[np.arange(3), f_i_ind[:, 0]] /
                         np.fft.rfft(qp)[np.arange(3), f_q_ind[:, 0]])  # for arcsine branch
        gamma = np.arcsin(np.sign(ratio) * 2 * np.mean(qp * ip, axis=-1) / (alpha * amp**2)) + np.pi * (ratio < 0)
        # choose the calibration with the highest transmission
        center = int(np.round((len(z) - 1) / 2))
        index = np.argmax([np.abs(z)[0], np.abs(z)[center], np.abs(z)[-1]])
        alpha = alpha[index]
        gamma = gamma[index]
    else:
        alpha = 1
        gamma = 0
    z = mixer_inverse((alpha, gamma, mixer_offset), z)
    # compute the magnitude and phase of the scattering parameter
    magnitude = np.abs(z)
    phase = np.unwrap(np.arctan2(z.imag, z.real))
    # filter the magnitude and phase if requested
    if use_filter:
        if filter_length is None:
            filter_length = int(np.round(len(magnitude) / 100.0))
        if filter_length % 2 == 0:
            filter_length += 1
        if filter_length < 3:
            filter_length = 3
        magnitude = sps.savgol_filter(magnitude, filter_length, 1)
        phase = sps.savgol_filter(phase, filter_length, 1)

    # calculate useful indices
    f_index_end = len(f) - 1  # last frequency index
    f_index_5pc = int(len(f) * 0.05)  # end of first 5% of data
    f_index_center = int(np.round(f_index_end / 2))  # center index
    # set up a unitless, reduced, midpoint frequency for baselines
    f_midpoint = f[f_index_center]  # frequency at the center of the data

    def ffm(fx):
        return (fx - f_midpoint) / f_midpoint

    # get the magnitude and phase data to fit
    mag_ends = np.concatenate((magnitude[0:f_index_5pc], magnitude[-f_index_5pc:-1]))
    phase_ends = np.concatenate((phase[0:f_index_5pc], phase[-f_index_5pc:-1]))
    freq_ends = ffm(np.concatenate((f[0:f_index_5pc], f[-f_index_5pc:-1])))
    # calculate the gain polynomials
    gain_poly = np.polyfit(freq_ends, mag_ends, 2 if quadratic_gain else 1)
    if not quadratic_gain:
        gain_poly = np.concatenate(([0], gain_poly))
    phase_poly = np.polyfit(freq_ends, phase_ends, 2 if quadratic_phase else 1)
    if not quadratic_phase:
        phase_poly = np.concatenate(([0], phase_poly))

    # guess f0
    f_index_min = np.argmin(magnitude - np.polyval(gain_poly, ffm(f)))
    f0_guess = f[f_index_min]
    # set some bounds (resonant frequency should not be within 5% of file end)
    f_min = min(f[f_index_5pc],  f[f_index_end - f_index_5pc])
    f_max = max(f[f_index_5pc],  f[f_index_end - f_index_5pc])
    if not f_min < f0_guess < f_max:
        f0_guess = f[f_index_center]

    # guess Q values
    mag_max = np.polyval(gain_poly, ffm(f[f_index_min]))
    mag_min = magnitude[f_index_min]
    fwhm = np.sqrt((mag_max**2 + mag_min**2) / 2.)  # fwhm is for power not amplitude
    fwhm_mask = magnitude < fwhm
    bandwidth = np.abs(f[fwhm_mask][-1] - f[fwhm_mask][0])
    # Q0 = f0 / fwhm bandwidth
    q0_guess = f0_guess / bandwidth
    # Q0 / Qi = min(mag) / max(mag)
    qi_guess = q0_guess * mag_max / mag_min
    # 1 / Q0 = 1 / Qc + 1 / Qi
    qc_guess = 1. / (1. / q0_guess - 1. / qi_guess)

    # make the parameters object (coerce all values to float to avoid ints and numpy types)
    params = lm.Parameters()
    # resonance parameters
    params.add('df', value=float(0), vary=fit_resonance)
    params.add('f0', value=float(f0_guess), min=f_min, max=f_max, vary=fit_resonance)
    params.add('qc', value=float(qc_guess), min=1, max=10**8, vary=fit_resonance)
    params.add('qi', value=float(qi_guess), min=1, max=10**8, vary=fit_resonance)
    params.add('a', value=float(0), min=0, max=2, vary=nonlinear_resonance and fit_resonance)  # a > 0.77 bifurcation
    # polynomial gain parameters
    params.add('gain0', value=float(gain_poly[2]), min=0, vary=fit_gain)
    params.add('gain1', value=float(gain_poly[1]), vary=fit_gain)
    params.add('gain2', value=float(gain_poly[0]), vary=quadratic_gain and fit_gain)
    # polynomial phase parameters
    params.add('phase0', value=float(phase_poly[2]), vary=fit_phase)
    params.add('phase1', value=float(phase_poly[1]), vary=fit_phase)
    params.add('phase2', value=float(phase_poly[0]), vary=quadratic_phase and fit_phase)
    # IQ mixer parameters
    params.add('i_offset', value=float(mixer_offset.real), vary=fit_offset)
    params.add('q_offset', value=float(mixer_offset.imag), vary=fit_offset)
    params.add('alpha', value=float(alpha), vary=fit_imbalance)
    params.add('gamma', value=float(gamma), min=gamma - np.pi / 2, max=gamma + np.pi / 2, vary=fit_imbalance)

    return params
