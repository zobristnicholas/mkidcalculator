import numpy as np
from scipy.signal import find_peaks, detrend

from mkidcalculator.io import Loop
from mkidcalculator.models import S21
from mkidcalculator.experiments.loop_fitting import basic_fit


def _integer_bandwidth(f, df):
    return int(np.round(df / (f[1] - f[0]) / 4) * 4)  # nearest even number divisible by 4


def find_resonators(f, magnitude, df, **kwargs):
    """
    Find resonators in a |S21| trace.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to the magnitude array.
        magnitude: numpy.ndarray
            The values corresponding to |S21|.
        df: float
            The frequency bandwidth for each resonator. df / 4 will be used as
            the max peak width unless overridden.
        kwargs: optional keyword arguments
            Optional keyword arguments to scipy.signal.find_peaks. Values here
            will override the defaults.
    Returns:
        peaks: numpy.ndarray, dtype=integer
            An array of peak integers
    """
    # resonator bandwidth in indices
    dfii = _integer_bandwidth(f, df)
    # detrend magnitude data for peak finding
    magnitude = detrend(magnitude)
    fit = np.argsort(magnitude)[:int(3 * len(magnitude) / 4):-1]
    poly = np.polyfit(f[fit], magnitude[fit], 1)
    magnitude = magnitude - np.polyval(poly, f)
    # find peaks
    kws = {"prominence": 1, "height": 5, "width": (None, int(dfii / 4))}
    kws.update(kwargs)
    peaks, _ = find_peaks(-magnitude, **kwargs)
    # cut out resonators that are separated from neighbors by less than df / 2
    right = np.hstack((np.diff(f[peaks]) > df / 2, False))
    left = np.hstack((False, np.diff(f[peaks][::-1])[::-1] < -df / 2))
    logic = left & right
    peaks = peaks[logic]
    return peaks


def collect_resonances(f, i, q, peaks, df):
    """
    Collect all of the resonances from a widesweep into an array.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to i and q.
        i: numpy.ndarray
            The I component of S21.
        q: numpy.ndarray
            The Q component of S21.
        peaks: numpy.ndarray, dtype=integer
            The indices corresponding to the resonator locations.
        df: float
            The final bandwidth of all of the outputs.
    Returns:
        f_array: numpy.ndarray
            A MxN array for the frequencies where M is the number of resonators
            and N is the number of frequencies.
        i_array: numpy.ndarray
            A MxN array for the I data where M is the number of resonators
            and N is the number of frequencies.
        q_array: numpy.ndarray
            A MxN array for the Q data where M is the number of resonators
            and N is the number of frequencies.
        peaks: numpy.ndarray, dtype=integer
            The peak indices corresponding to resonator locations. Some indices
            may be removed due to encroaching nearby resonators.
    """
    # resonator bandwidth in indices
    dfii = _integer_bandwidth(f, df)
    # collect resonance data into arrays
    f_array = np.empty((len(peaks), int(dfii / 2)))
    i_array = np.empty(f_array.shape)
    q_array = np.empty(f_array.shape)
    for ii in range(f_array.shape[0]):
        f_array[ii, :] = f[int(peaks[ii] - dfii / 4): int(peaks[ii] + dfii / 4)]
        i_array[ii, :] = i[int(peaks[ii] - dfii / 4): int(peaks[ii] + dfii / 4)]
        q_array[ii, :] = q[int(peaks[ii] - dfii / 4): int(peaks[ii] + dfii / 4)]
    # cut out resonators that aren't centered (large resonator tails on either side)
    logic = np.argmin(i_array ** 2 + q_array ** 2, axis=-1) == dfii / 4
    return f_array[logic, :], i_array[logic, :], q_array[logic, :], peaks[logic]


def widesweep_fit(f, i, q, df, fit_type=basic_fit, find_resonators_kwargs=None, loop_kwargs=None, **kwargs):
    """
    Fits each resonator in the widesweep.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to i and q.
        i: numpy.ndarray
            The I component of S21.
        q: numpy.ndarray
            The Q component of S21.
        df: float
            The frequency bandwidth over which to perform the fit.
        fit_type: function (optional)
            A function that takes a mkidcalculator.io.Loop as the first
            argument and returns the fitted loop.
        find_resonators_kwargs: dictionary (optional)
            A dictionary of options for the find_resonators function.
        loop_kwargs: dictionary (optional)
            A dictionary of options for loading the loop with
            Loop.from_python().
        kwargs: optional keyword arguments
            Optional keyword arguments to give to the fit_type function.
    Returns:
        loops: A list of mkidcalculator.Loop objects
            The loop objects that were fit.
    """
    # prepare the data
    peaks = find_resonators(f, 10 * np.log10(i**2 + q**2), df, **kwargs)
    f_array, i_array, q_array, _ = collect_resonances(f, i, q, peaks, df)
    # set up the loop kwargs
    kws = {"attenuation": 0., "field": 0., "temperature": 0.}
    if loop_kwargs is not None:
        kws.update(loop_kwargs)
    # fit the loops
    loops = []
    for ii in range(f_array.shape[0]):
        loop = Loop.from_python(i + 1j * q, f, **kws)
        loops.append(fit_type(loop, **kwargs))

