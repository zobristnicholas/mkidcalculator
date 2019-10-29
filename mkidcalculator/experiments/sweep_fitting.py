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

# TODO: remove file .. no longer needed with Sweep.from_widesweep()


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
    Collect all of the resonances from a sweep into an array.
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


def sweep_fit(f, z, df, fit_type=basic_fit, find_resonators_kwargs=None, loop_kwargs=None, **kwargs):
    """
    Fits each resonator in the sweep.
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


def get_loop_fit_info(loops, label='best', parameters=("fr", "qi", "q0", "chi2"), bounds=None, errorbars=None,
                      success=None):
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
        bounds: tuple of numbers or tuples
            The bounds for the parameters. It must be a tuple of the same
            length as the parameters keyword argument. Each element is either
            an upper bound on the parameter or a tuple, e.g. (lower bound,
            upper bound). None can be used as a placeholder to skip a
            parameter. The default is None and no bounds are used.
        errorbars: boolean
            If errorbars is True, only data from loop fits that could compute
            errorbars on the fit parameters is included. If errorbars is False,
            only data from loop fits that could not compute errorbars on the
            fit parameters is included. The default is None, and no filtering
            on the errorbars is done.
        success: boolean
            If success is True, only data from successful loop fits is
            included. If False, only data from failed loop fits is
            included. The default is None, and no filtering on fit success is
            done. Note: fit success is typically a bad indicator on fit
            quality. It only ever fails when something really bad happens.
    Returns:
        outputs: tuple of numpy.ndarray objects
            The outputs in the same order as parameters.
    """
    outputs = []
    for parameter in parameters:
        outputs.append([])
        for loop in loops:
            result = loop.lmfit_results[label]['result']
            if errorbars is not None and result.errorbars != errorbars:
                continue  # skip if wrong errorbars setting
            if success is not None and result.success != success:
                continue  # skip if wrong success setting
            if parameter == "chi2":
                outputs[-1].append(result.redchi)
            else:
                outputs[-1].append(result.params[parameter].value)
    # turn outputs into a list of numpy arrays
    for index, output in enumerate(outputs):
        outputs[index] = np.array(output)
    # format bounds if None
    if bounds is None:
        bounds = [None] * len(outputs)
    # make filtering logic
    logic = np.ones(outputs[-1].shape, dtype=bool)
    for index, output in enumerate(outputs):
        if bounds[index] is None:
            continue
        elif isinstance(bounds[index], (list, tuple)):
            logic = logic & (output >= bounds[index][0]) & (output <= bounds[index][1])
        else:
            logic = logic & (output <= bounds[index])
    # filter outputs
    for index, output in enumerate(outputs):
        outputs[index] = output[logic]
    return tuple(outputs)


def plot_parameter_hist(parameter, title=None, x_label=True, y_label=True, label_kwargs=None, tick_kwargs=None,
                        tighten=True, return_bin=False, axes=None, **kwargs):
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
        label_kwargs: dictionary
            Keyword arguments for the axes labels in axes.set_*label(). The
            default is None which uses default options. Keywords in this
            dictionary override the default options.
        tick_kwargs: dictionary
            Keyword arguments for the ticks using axes.tick_params(). The
            default is None which uses the default options. Keywords in
            this dictionary override the default options.
        tighten: boolean (optional)
            Whether or not to apply figure.tight_layout() at the end of the
            plot. The default is True.
        return_bin: boolean (optional)
            Whether or not to include the binned information in the returned
            values. The default is False and only the axes are returned.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
        kwargs: optional keyword arguments
            Extra keyword arguments to send to axes.hist().
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted data.
            centers: numpy.ndarray
                The histogram bin centers. Only returned if return_bin is True.
            counts: numpy.ndarray
                The histogram bin counts. Only returned if return_bin is True.
    """
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    kws = {"bins": 50}
    if kwargs:
        kws.update(kwargs)
    counts, edges, _ = axes.hist(parameter, **kws)
    axes.set_xlim(left=0)
    kws = {}
    if label_kwargs is not None:
        kws.update(label_kwargs)
    if x_label is not False:
        axes.set_xlabel("parameter values" if x_label is True else x_label, **kws)
    if y_label is not False:
        axes.set_ylabel("counts per bin" if y_label is True else y_label, **kws)
    if tick_kwargs is not None:
        axes.tick_params(**tick_kwargs)
    if title is not False and title is not None:
        axes.set_title(title)
    if tighten:
        figure.tight_layout()
    if return_bin:
        centers = 0.5 * (edges[1:] + edges[:-1])
        return axes, centers, counts
    return axes


def plot_parameter_vs_f(parameter, f, title=None, x_label=True, y_label=True, label_kwargs=None, tick_kwargs=None,
                        tighten=True, bins=30, extend=True, return_bin=False, axes=None, **kwargs):
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
        label_kwargs: dictionary
            Keyword arguments for the axes labels in axes.set_*label(). The
            default is None which uses default options. Keywords in this
            dictionary override the default options.
        tick_kwargs: dictionary
            Keyword arguments for the ticks using axes.tick_params(). The
            default is None which uses the default options. Keywords in
            this dictionary override the default options.
        tighten: boolean (optional)
            Whether or not to apply figure.tight_layout() at the end of the
            plot. The default is True.
        bins: integer (optional)
            The number of bins to use in the plot. The default is 30.
        extend: boolean (optional)
            Determines whether or not to extend the data so that there is a
            bin with zero values on either side of the frequency range. The
            default is True.
        return_bin: boolean (optional)
            Whether or not to include the binned information in the returned
            values. The default is False and only the axes are returned.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
        kwargs: optional keyword arguments
            Extra keyword arguments to send to axes.step().
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted data.
            centers: numpy.ndarray
                The bin centers. Only returned if return_bin is True.
            medians: numpy.ndarray
                The median values in each bin. Only returned if return_bin is
                True.
    """
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    kws = {"where": "mid"}
    if kwargs:
        kws.update(kwargs)
    edges = np.linspace(f.min(), f.max(), bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    medians = np.empty(bins - 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # median of empty slice
        for ii in range(len(medians) - 1):
            medians[ii] = np.median(parameter[(f >= edges[ii]) & (f < edges[ii + 1])])
        medians[-1] = np.median(parameter[(f >= edges[-2]) & (f <= edges[-1])])  # last bin is fully closed
    medians[np.isnan(medians)] = 0
    if extend:
        dx = centers[1] - centers[0]
        centers = np.hstack([centers[0] - dx, centers, centers[-1] + dx])
        medians = np.hstack([0, medians, 0])
    axes.step(centers, medians, **kws)
    axes.set_xlim(centers.min(), centers.max())
    axes.set_ylim(bottom=0)
    kws = {}
    if label_kwargs is not None:
        kws.update(label_kwargs)
    if x_label is not False:
        axes.set_xlabel("frequency [GHz]" if x_label is True else x_label, **kws)
    if y_label is not False:
        axes.set_ylabel("median parameter" if y_label is True else y_label, **kws)
    if tick_kwargs is not None:
        axes.tick_params(**tick_kwargs)
    if title and title is not None:
        axes.set_title(title)
    if tighten:
        figure.tight_layout()
    if return_bin:
        return axes, centers, medians
    return axes


def plot_sweep_summary(loops, qi_cutoff=None, q0_cutoff=None, chi2_cutoff=None, success=True, errorbars=True,
                       title=True, tighten=True, label='best', plot_kwargs=None, figure=None):
    """
    Plot a summary of a sweep fit.
    Args:
        loops: list of mkidcalculator.Loop objects
            The fitted loops to use for the summary plot
        qi_cutoff: float (optional)
            The maximum Qi to use in the summary plot. The default is
            None and no cutoff is used. A tuple may be used to enforce a lower
            bound.
        q0_cutoff: float (optional)
            The maximum Q0 to use in the summary plot. The default is
            None and no cutoff is used. A tuple may be used to enforce a lower
            bound.
        chi2_cutoff: float (optional)
            The maximum chi2 to use in the summary plot. The default is
            None and no cutoff is used. A tuple may be used to enforce a lower
            bound.
        errorbars: boolean
            If errorbars is True, only data from loop fits that could compute
            errorbars on the fit parameters is included. If errorbars is False,
            only data from loop fits that could not compute errorbars on the
            fit parameters is included. The default is True. None may be used
            to enforce no filtering on the errorbars.
        success: boolean
            If success is True, only data from successful loop fits is
            included. If False, only data from failed loop fits is
            included. The default is True. None may be used
            to enforce no filtering on success. Note: fit success is typically
            a bad indicator on fit quality. It only ever fails when something
            really bad happens.
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
    fr, qi, q0, chi2 = get_loop_fit_info(loops, label=label, parameters=("fr", "qi", "q0", "chi2"), success=success,
                                         bounds=(None, qi_cutoff, q0_cutoff, chi2_cutoff), errorbars=errorbars)

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
    kws = {"x_label": "$Q_i$"}
    if plot_kwargs[0]:
        kws.update(plot_kwargs[0])
    plot_parameter_hist(qi, axes=axes_list[0], **kws)
    kws = {"y_label": "median $Q_i$"}
    if plot_kwargs[1]:
        kws.update(plot_kwargs[1])
    plot_parameter_vs_f(qi, fr, axes=axes_list[1], **kws)
    kws = {"x_label": "$Q_0$"}
    if plot_kwargs[2]:
        kws.update(plot_kwargs[2])
    plot_parameter_hist(q0, axes=axes_list[2], **kws)
    kws = {"y_label": "median $Q_0$"}
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
        title = "sweep fit summary: '{}'".format(label) if title is True else title
        figure.suptitle(title, fontsize=15)
        rect = [0, 0, 1, .95]
    else:
        rect = [0, 0, 1, 1]
    # tighten
    if tighten:
        figure.tight_layout(rect=rect)
    return figure
