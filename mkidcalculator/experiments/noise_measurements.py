import numpy as np
import scipy.stats as stats
import scipy.constants as sc


def compute_noise_numbers(f, t_hot, t_cold, s0, s1, s2=None, s3=None, gp=None, lh=1, lp=1, z0=50, summarize=False):
    """
    Compute the noise numbers of the different components of a MKID readout.
    This code follows the layout presented by Zobrist et al. 2019 in appendix A
    and B of the supplementary material (doi.org/10.1063/1.5098469).
    Args:
        f: float
            Frequency in Hz of the readout tone.
        t_hot: float
            Temperature in Kelvin of the highest temperature calibration
            attenuator.
        t_cold: float
            Temperature in Kelvin of the lowest temperature calibration
            attenuator.
        s0: numpy.ndarray
            PSD_II + PSD_QQ when the HEMT input is terminated at the t_hot
            attenuator.
        s1: numpy.ndarray
            PSD_II + PSD_QQ when the HEMT input is terminated at the t_cold
            attenuator.
        s2: numpy.ndarray (optional)
            PSD_II + PSD_QQ when the HEMT input is connected to an MKID. If
            an MKID is connected to the same switch as the calibration
            attenuators, this noise spectrum can be provided to calculate
            the input noise. The default is None, which assumes no MKID is
            connected.
        s3: numpy.ndarray (optional)
            PSD_II + PSD_QQ when the HEMT input is connected to an MKID and a
            parametric amplifier is turned on. If a parametric amplifier (or
            other amplifier that has unity gain when off) is between the MKID
            and the same switch as the calibration attenuators, this noise
            spectrum can be provided to calculate the parametric amplifier noise.
            The default is None, which assumes no parametric amplifier is
            connected. s2 and gp must be provided if this parameter is used.
        gp: float (optional)
            The parametric amplifier power gain (not in units of dB). s2 and s3
            must be provided if this parameter is used. The default is None,
            which assumes no parametric amplifier is connected.
        lh: float (optional)
            The multiplicative power loss between the switch and HEMT. The
            default is 1 and no loss is assumed.
        lp: float (optional)
            The multiplicative power loss between the parametric amplifier and
            the switch. The default is 1 and no loss is assumed.
        z0: float (optional)
            The characteristic impedance of the transmission line in Ohms. The
            default is 50 Ohms.
        summarize: boolean (optional)
            Print a summary of the calculation. The default is False and
            nothing is printed.
    Returns:
        results: dictionary
            A dictionary containing the results of the calculation. Some
            key/value pairs will not be included if all of the keyword
            arguments are not used. The possible key/value pairs are listed
            below.
            Always present:
                gh: np.ndarray
                    The HEMT amplifier gain (including all of the losses and
                    gains after the HEMT).
                sh: np.ndarray
                    The HEMT amplifier noise power spectral density.
                ah: np.ndarray
                    The HEMT amplifier noise number in units of quanta.
            Included if s2 is used:
                si: np.ndarray
                    The input noise power spectral density.
                ai: np.ndarray
                    The input noise number in units of quanta.
                gh_sys: np.ndarray
                    The system gain including specified losses before the HEMT
                    amplifier, lh (and including all of the losses and gains
                    after the HEMT amplifier).
                ah_sys: np.ndarray
                    The system noise number corresponding to the specified
                    system power spectral density, s2.
            Included if s2, s3, and gp are used:
                sp: np.ndarray
                    The parametric amplifier power spectral density.
                ap: np.ndarray
                    The parametric amplifier noise number in units of quanta.
                gp_sys: np.ndarray
                    The system gain with the parametric amplifier including
                    specified losses before the HEMT amplifier, lh, and
                    specified losses after the parametric amplifier, lp (and
                    including all of the losses and gains after the HEMT
                    amplifier).
                ap_sys: np.ndarray
                    The system noise number with the parametric amplifier in
                    units of quanta.
    """
    # check inputs
    if (s3 is not None or gp is not None) and (s2 is None or s3 is None or gp is None):
        raise ValueError("'s2', 's3', and 'gp' must be provided if either 's2' or 'gp' is provided")
    # calibration parameters
    s_hot = 2 * sc.h * f * z0 / np.tanh(sc.h * f / (2 * sc.k * t_hot))
    s_cold = 2 * sc.h * f * z0 / np.tanh(sc.h * f / (2 * sc.k * t_cold))
    # HEMT gain and noise
    gh = (s0 - s1) / (s_hot - s_cold) * lh
    sh = (s1 * s_hot - s0 * s_cold) / (s0 - s1)
    ah = sh / (4 * sc.h * f * z0)
    results = {'gh': gh, 'sh': sh, 'ah': ah}
    if summarize:
        loc, interval = noise_estimate(ah, prior=0.5, n_sigma=1.)
        noise_report(loc, label="HEMT noise number: ", statistical=interval, return_string=False)
    # Input noise
    if s2 is not None:
        si = ((s2 - s1) * s_hot + (s0 - s2) * s_cold) / (s0 - s1) * lp
        ai = si / (4 * sc.h * f * z0)
        gh_sys = gh / lh  # system gain
        ah_sys = s2 / (4 * sc.h * f * z0 * gh_sys)  # total system noise
        results.update({'si': si, 'ai': ai, 'gh_sys': gh_sys, 'ah_sys': ah_sys})
        if summarize:
            loc, interval = noise_estimate(ai, prior=0.5, n_sigma=1.)
            noise_report(loc, label="Input noise number: ", statistical=interval, return_string=False)
            loc, interval = noise_estimate(ah_sys, prior=0.5, n_sigma=1.)
            noise_report(loc, label="System noise number: ", statistical=interval, return_string=False)
    # para-amp noise
    if gp is not None:
        sp = (((s3 - s1 - gp * (s2 - s1)) * s_hot +
               (s0 - s3 - gp * (s0 - s2)) * s_cold) / (gp * (s0 - s1))) * lp
        ap = sp / (4 * sc.h * f * z0)
        gp_sys = gh / lh * gp / lp  # system gain with para-amp on
        ap_sys = s2 / (4 * sc.h * f * z0 * gp_sys)  # total system noise with para-amp on
        results.update({'sp': sp, 'ap': ap, 'gp_sys': gp_sys, 'ap_sys': ap_sys})
        if summarize:
            loc, interval = noise_estimate(ap, prior=0.5, n_sigma=1.)
            noise_report(loc, label="Para-amp noise number: ", statistical=interval, return_string=False)
            loc, interval = noise_estimate(ap_sys, prior=0.5, n_sigma=1.)
            noise_report(loc, label="Para-amp system noise number: ", statistical=interval, return_string=False)
    return results


def noise_estimate(noise, prior=0., n_sigma=1.):
    """
    Compute an estimate of the average noise and a confidence interval from
    multiple measurements.
    Args:
        noise: np.ndarray
            Multiple measurements of the noise.
        prior: float (optional)
            A lower bound on the possible values of the result. Standard noise
            cannot be below 0, for example. Amplifier noise in units of quanta
            cannot be below 0.5. The default is 0. To disable the prior, use
            None.
        n_sigma: float (optional)
            The number of sigma to use for the confidence interval. The sigma
            number corresponds to the area under the posterior distribution in
            the returned interval that is equivalent to the area under a
            gaussian distribution between -sigma and +sigma.
    Returns:
        loc: float
            The noise estimate.
        interval: tuple of floats
            The potentially asymmetric confidence interval.
    """
    if prior is None:
        prior = -np.inf
    n = noise.size  # number of data points
    loc = np.mean(noise)  # estimate of mean
    scale = np.std(noise, ddof=1) / np.sqrt(n)  # estimate of gaussian standard deviation
    p_sigma = 1 - 2 * stats.norm.cdf(-n_sigma)  # gaussian probability between -n_sigma and n_sigma
    df = n - 1  # degrees of freedom
    # posterior probability that the result is less than the prior (1/2 for amplifiers, 0 otherwise)
    less_than = stats.t.cdf(prior, df, loc, scale)
    # compute the interval corresponding to the sigma level adjusted for leaving out some area less than the prior
    error = stats.t.ppf(p_sigma * (1 - less_than) + less_than, df, loc, scale)
    interval = (max(prior, loc - error), loc + error)
    return loc, interval


def noise_report(loc, label="", statistical=None, systematic=None, return_string=False):
    """
    Print a string summarizing the noise value.
    Args:
        loc: float
            The mean noise value.
        label: string (optional)
            An optional label to prepend to the print statement. The default is
            an empty string.
        statistical: tuple of floats (optional)
            The statistical confidence interval. The default is None and no
            interval is used.
        systematic: tuple of floats (optional)
            The systematic confidence interval. The default is None and no
            interval is used.
        return_string: boolean
            Determines if the string is returned or printed. The default is
            False and the string is printed.
    Returns:
        string: string
            A string containing the fit report. None is output if
            return_string is False.
    """
    loc_string = "{:.3f}"
    statistical_string = ", (+{:.3f}, -{:.3f}) statistical"
    systematic_string = ", (+{:.3f}, -{:.3f}) systematic"
    string = label + loc_string.format(loc)
    if statistical is not None:
        string += statistical_string.format(statistical[0], statistical[1])
    if systematic is not None:
        string += systematic_string.format(systematic[0], systematic[1])
    if return_string:
        return string
    else:
        print(string)