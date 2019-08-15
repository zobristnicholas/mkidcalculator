import inspect
import logging
import warnings
import numpy as np
from matplotlib import gridspec
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


def get_loop_fit_info(loops, label='best', parameters=("fr", "qi", "q0", "chi2")):
    """
    Get fit information from a list of Loops
    Args:
        loops: list of mkidcalculator.Loop objects
            The fitted loops to extract the information from.
        label: string (optional)
            The fit label to use.
        parameters: tuple of strings
            The fit parameters to report. "chi2" can be used to retrieve
            the reduced chi squared values.
    Returns:
        outputs: tuple of numpy.ndarray objects
            The outputs in the same order as parameters.
    """
    outputs = []
    for parameter in parameters:
        outputs.append([])
        for loop in loops:
            if parameter == "chi2":
                outputs[-1].append(loop.lmfit_results[label]['result'].redchi)
            else:
                outputs[-1].append(loop.lmfit_results[label]['result'].params[parameter].value)
    for index, output in enumerate(outputs):
        outputs[index] = np.array(output)
    return tuple(outputs)


def plot_parameter_hist(parameter, title=None, x_label=True, y_label=True, tighten=True, axes=None, **kwargs):
    """
    Plot a parameter histogram.
    Args:
        parameter: numpy.ndarray
            An array of parameter values.
        title: string (optional)
            The title for the plot. The default is None and no title is made.
        x_label: string, boolean (optional)
            The x label for the plot. The default is True and the default label
            is used. If False, no label is used.
        y_label: string, boolean (optional)
            The y label for the plot. The default is True and the default label
            is used. If False, no label is used.
        tighten: boolean (optional)
            Whether or not to apply figure.tight_layout() at the end of the
            plot. The default is True.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
        kwargs: optional keyword arguments
            Extra keyword arguments to send to axes.hist().
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted data.
    """
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    kws = {"bins": 50}
    if kwargs:
        kws.update(kwargs)
    axes.hist(parameter, **kws)
    axes.set_xlim(left=0)
    if x_label is not False:
        axes.set_xlabel("parameter values" if x_label is True else x_label)
    if y_label is not False:
        axes.set_ylabel("counts per bin" if y_label is True else y_label)
    if title is not False and title is not None:
        axes.set_title(title)
    if tighten:
        figure.tight_layout()
    return axes


def plot_parameter_vs_f(parameter, f, title=None, x_label=True, y_label=True, tighten=True, bins=30, axes=None,
                        **kwargs):
    """
    Plot a parameter vs frequency.
    Args:
        parameter: numpy.ndarray
            An array of parameter values.
        f: numpy.ndarray
            The frequencies corresponding to the parameter values.
        title: string (optional)
            The title for the plot. The default is None and no title is made.
        x_label: string, boolean (optional)
            The x label for the plot. The default is True and the default label
            is used. If False, no label is used.
        y_label: string, boolean (optional)
            The y label for the plot. The default is True and the default label
            is used. If False, no label is used.
        tighten: boolean (optional)
            Whether or not to apply figure.tight_layout() at the end of the
            plot. The default is True.
        bins: integer (optional)
            The number of bins to use in the plot. The default is 30.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
        kwargs: optional keyword arguments
            Extra keyword arguments to send to axes.step().
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted data.
    """
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    kws = {"where": "mid"}
    if kwargs:
        kws.update(kwargs)
    bin_edges = np.linspace(f.min(), f.max(), bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    medians = np.empty(bins - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # median of empty slice
        for ii in range(len(medians) - 1):
            medians[ii] = np.median(parameter[(f >= bin_edges[ii]) & (f < bin_edges[ii + 1])])
        medians[-1] = np.median(parameter[(f >= bin_edges[-2]) & (f <= bin_edges[-1])])  # last bin is fully closed
    medians[np.isnan(medians)] = 0
    axes.step(bin_centers, medians, **kws)
    axes.set_xlim(bin_centers.min(), bin_centers.max())
    axes.set_ylim(bottom=0)
    if x_label is not False:
        axes.set_xlabel("frequency [GHz]" if x_label is True else x_label)
    if y_label is not False:
        axes.set_ylabel("median parameter" if y_label is True else y_label)
    if title and title is not None:
        axes.set_title(title)
    if tighten:
        figure.tight_layout()
    return axes


def plot_widesweep_summary(loops, qi_cutoff=np.inf, q0_cutoff=np.inf, chi2_cutoff=np.inf, title=True, tighten=True,
                           label='best', plot_kwargs=None, figure=None):
    """
    Plot a summary of a widesweep fit.
    Args:
        loops: list of mkidcalculator.Loop objects
            The fitted loops to use for the summary plot
        qi_cutoff: float (optional)
            The maximum Qi to use in the summary plot. The default is
            numpy.inf.
        q0_cutoff: float (optional)
            The maximum Q0 to use in the summary plot. The default is
            numpy.inf.
        chi2_cutoff: float (optional)
            The maximum chi2 to use in the summary plot. The default is
            numpy.inf.
        title: string or boolean (optional)
            The title to use for the summary plot. The default is True and the
            default title will be applied. If False, no title is applied.
        tighten: boolean (optional)
            Whether or not to apply figure.tight_layout() at the end of the
            plot. The default is True.
        label: string (optional)
            The fit label to use for the plots. The default is 'best'.
        plot_kwargs: dictionary or list of dictionaries (optional)
            A dictionary or list of dictionaries containing plot options. If
            only one is provided, it is used for all of the plots. If a list
            is provided, it must be of the same length as the number of plots.
            No kwargs are passed by default.
        figure: matplotlib.figure.Figure class
            A figure to use for the plots. The default is None and one is
            created automatically.
    Returns:
        figure: matplotlib.figure.Figure class
            The figure object for the plot.
    """
    # get the data
    fr, qi, q0, chi2 = get_loop_fit_info(loops, label=label)
    logic = (qi <= qi_cutoff) & (q0 <= q0_cutoff) & (chi2 <= chi2_cutoff)
    fr = fr[logic]
    qi = qi[logic]
    q0 = q0[logic]
    chi2 = chi2[logic]

    # create figure if needed
    if figure is None:
        from matplotlib import pyplot as plt
        figure = plt.figure(figsize=(8.5, 11))
    # setup figure axes
    gs = gridspec.GridSpec(3, 2)
    axes_list = np.array([figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]),
                          figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1]),
                          figure.add_subplot(gs[2, 0]), figure.add_subplot(gs[2, 1])])
    # check plot kwargs
    if plot_kwargs is None:
        plot_kwargs = {}
    if isinstance(plot_kwargs, dict):
        plot_kwargs = [plot_kwargs] * len(axes_list)
    # add plots
    kws = {"x_label": "Q$_i$"}
    if plot_kwargs[0]:
        kws.update(plot_kwargs[0])
    plot_parameter_hist(qi, axes=axes_list[0], **kws)
    kws = {"y_label": "median Q$_i$"}
    if plot_kwargs[1]:
        kws.update(plot_kwargs[1])
    plot_parameter_vs_f(qi, fr, axes=axes_list[1], **kws)
    kws = {"x_label": "Q$_0$"}
    if plot_kwargs[2]:
        kws.update(plot_kwargs[2])
    plot_parameter_hist(q0, axes=axes_list[2], **kws)
    kws = {"y_label": "median Q$_0$"}
    if plot_kwargs[3]:
        kws.update(plot_kwargs[3])
    plot_parameter_vs_f(q0, fr, axes=axes_list[3], **kws)
    kws = {"x_label": r"$\chi^2 / \nu$"}
    if plot_kwargs[3]:
        kws.update(plot_kwargs[4])
    plot_parameter_hist(chi2, axes=axes_list[4], **kws)
    kws = {"y_label": r"median $\chi^2 / \nu$"}
    if plot_kwargs[3]:
        kws.update(plot_kwargs[5])
    plot_parameter_vs_f(chi2, fr, axes=axes_list[5], **kws)
    # add title
    if title:
        title = "widesweep fit summary: '{}'".format(label) if title is True else title
        figure.suptitle(title, fontsize=15)
        rect = [0, 0, 1, .95]
    else:
        rect = [0, 0, 1, 1]
    # tighten
    if tighten:
        figure.tight_layout(rect=rect)
    return figure
