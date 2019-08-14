import inspect
import logging
import numpy as np
from scipy.signal import find_peaks, detrend

from mkidcalculator.io.loop import Loop
from mkidcalculator.models import S21
from mkidcalculator.experiments.loop_fitting import basic_fit

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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
    peaks, _ = find_peaks(-magnitude, **kws)
    # cut out resonators that are separated from neighbors by less than df / 2
    right = np.hstack((np.diff(f[peaks]) > df / 2, False))
    left = np.hstack((False, np.diff(f[peaks][::-1])[::-1] < -df / 2))
    logic = left & right
    peaks = peaks[logic]
    return peaks


def collect_resonances(f, z, peaks, df):
    """
    Collect all of the resonances from a widesweep into an array.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to i and q.
        z: numpy.ndarray
            The S21 complex scattering data.
        peaks: numpy.ndarray, dtype=integer
            The indices corresponding to the resonator locations.
        df: float
            The final bandwidth of all of the outputs.
    Returns:
        f_array: numpy.ndarray
            A MxN array for the frequencies where M is the number of resonators
            and N is the number of frequencies.
        z_array: numpy.ndarray
            A MxN array for the S21 data where M is the number of resonators
            and N is the number of frequencies.
        peaks: numpy.ndarray, dtype=integer
            The peak indices corresponding to resonator locations. Some indices
            may be removed due to encroaching nearby resonators.
    """
    # resonator bandwidth in indices
    dfii = _integer_bandwidth(f, df)
    # collect resonance data into arrays
    f_array = np.empty((len(peaks), int(dfii / 2)))
    z_array = np.empty(f_array.shape, dtype=np.complex)
    for ii in range(f_array.shape[0]):
        f_array[ii, :] = f[int(peaks[ii] - dfii / 4): int(peaks[ii] + dfii / 4)]
        z_array[ii, :] = z[int(peaks[ii] - dfii / 4): int(peaks[ii] + dfii / 4)]
    # cut out resonators that aren't centered (large resonator tails on either side)
    logic = np.argmin(np.abs(z_array), axis=-1) == dfii / 4
    return f_array[logic, :], z_array[logic, :], peaks[logic]


def widesweep_fit(f, z, df, fit_type=basic_fit, find_resonators_kwargs=None, loop_kwargs=None, **kwargs):
    """
    Fits each resonator in the widesweep.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to i and q.
        z: numpy.ndarray
            The S21 complex scattering data.
        df: float
            The frequency bandwidth over which to perform the fit.
        fit_type: function (optional)
            A function that takes a mkidcalculator.Loop as the first
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
    ind = np.argsort(f)
    f = f[ind]
    z = z[ind]
    kws = {}
    if find_resonators_kwargs is not None:
        kws.update(find_resonators_kwargs)
    peaks = find_resonators(f, 20 * np.log10(np.abs(z)), df, **kws)
    f_array, z_array, _ = collect_resonances(f, z, peaks, df)
    # set up the loop kwargs
    kws = {"attenuation": 0., "field": 0., "temperature": 0.}
    if loop_kwargs is not None:
        kws.update(loop_kwargs)
    # get label for logging
    params = inspect.signature(fit_type).parameters
    if 'label' in kwargs.keys():
        label = kwargs['label']
    elif 'label' in params.keys() and isinstance(params['label'].default, str):
        label = inspect.signature(fit_type).parameters['label'].default
    else:
        label = 'best'
    # fit the loops
    loops = []
    for ii in range(f_array.shape[0]):
        loop = Loop.from_python(z_array[ii, :], f_array[ii, :], **kws)
        loops.append(fit_type(loop, **kwargs))
        result = loops[-1].lmfit_results[label]['result']
        log.info("'{}' fit {} completed with a reduced chi of {:g}".format(label, ii, result.redchi))
    return loops
