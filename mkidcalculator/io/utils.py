import os
import numbers
import logging
import tempfile
import numpy as np
import lmfit as lm
from collections import OrderedDict
import scipy.constants as c
from scipy.signal import find_peaks, detrend

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class NpzHolder:
    """Loads npz file when requested and saves them."""
    MAX_SIZE = 200

    def __init__(self):
        self._files = OrderedDict()

    def __getitem__(self, item):
        # if string load and save to cache
        if isinstance(item, str):
            item = os.path.abspath(item)
            # check if already loaded
            if item in self._files.keys():
                log.debug("loaded from cache: {}".format(item))
                return self._files[item]
            else:
                self._check_size()
                npz = np.load(item, allow_pickle=True)
                log.debug("loaded: {}".format(item))
                self._files[item] = npz
                log.debug("saved to cache: {}".format(item))
                return self._files[item]
        # if NpzFile skip loading but save if it hasn't been loaded before
        elif isinstance(item, np.lib.npyio.NpzFile):
            file_name = os.path.abspath(item.fid.name)
            if file_name not in _loaded_npz_files.keys():
                self._check_size()
                log.debug("loaded: {}".format(file_name))
                self._files[file_name] = item
                log.debug("saved to cache: {}".format(file_name))
            else:
                log.debug("loaded from cache: {}".format(file_name))
            return item
        elif item is None:
            return None
        else:
            raise ValueError("'item' must be a valid file name or a numpy npz file object.")

    def free_memory(self, file_names=None):
        """
        Removes file names in file_names from active memory. If file_names is
        None, all are removed (default).
        """
        if file_names is None:
            file_names = self._files.keys()
        elif isinstance(file_names, str):
            file_names = [file_names]
        for file_name in file_names:
            npz = self._files.pop(file_name, None)
            del npz

    def _check_size(self):
        for _ in range(max(len(self._files) - self.MAX_SIZE + 1, 0)):
            item = self._files.popitem(last=False)
            log.debug("Max cache size reached. Removed from cache: {}".format(item[0]))


_loaded_npz_files = NpzHolder()  # cache of already loaded files


def compute_phase_and_amplitude(cls, label="best", fit_type="lmfit", fr="fr", unwrap=True):
    """
    Compute the phase and amplitude traces stored in pulse.p_trace and
    pulse.a_trace.
    Args:
        cls: Pulse or Noise class
            The Pulse or Noise class used to create the phase and amplitude
            data.
        label: string
            Corresponds to the label in the loop.lmfit_results or
            loop.emcee_results dictionaries where the fit parameters are.
            The resulting DataFrame is stored in
            object.loop_parameters[label]. The default is "best", which gets
            the parameters from the best fits.
        fit_type: string
            The type of fit to use. Allowed options are "lmfit", "emcee",
            and "emcee_mle" where MLE estimates are used instead of the
            medians. The default is "lmfit".
        fr: string
            The parameter name that corresponds to the resonance frequency.
            The default is "fr" which gives the resonance frequency for the
            mkidcalculator.S21 model. This parameter determines the zero
            point for the traces.
        unwrap: boolean
            Determines whether or not to unwrap the phase data. The default
            is True.
    """
    # clear prior data
    cls.clear_traces()
    # get the model and parameters
    _, result_dict = cls.loop._get_model(fit_type, label)
    model = result_dict["model"]
    params = result_dict["result"].params
    # get the resonance frequency and loop center
    fr = params[fr].value
    # get complex IQ data for the traces and loop at the resonance frequency
    traces = cls.i_trace + 1j * cls.q_trace
    z_fr = model.model(params, fr)
    f = np.empty(traces.shape)
    f.fill(cls.f_bias)
    # calibrate the IQ data
    traces = model.calibrate(params, traces, f, center=True)
    z_fr = model.calibrate(params, z_fr, fr, center=True)  # should be real if no loop asymmetry
    # compute the phase and amplitude traces from the centered traces
    cls.p_trace = np.unwrap(np.angle(traces) - np.angle(z_fr)) if unwrap else np.angle(traces) - np.angle(z_fr)
    cls.a_trace = np.abs(traces) / np.abs(z_fr) - 1


def offload_data(cls, excluded_keys=(), npz_key="_npz", prefix="", directory_key="_directory"):
    """
    Offload data in excluded_keys from the class to an npz file. The npz file
    name is stored in cls.npz_key.
    Args:
        cls: class
            The class being unpickled
        excluded_keys: iterable of strings
            Keys to force into npz format. The underlying attributes must be
            numpy arrays. The default is to not exclude any keys.
        npz_key: string
            The class attribute name that corresponds to where the npz file was
            stored. The default is "_npz".
        prefix: string
            File name prefix for the class npz file if a new one needs to be
            made. The default is no prefix.
        directory_key: string
            The class attribute that corresponds to the data directory. If it
            doesn't exist, than the current directory is used.
    Returns:
        cls.__dict__: dictionary
            The new class dict which can be used for pickling.
    """
    # get the directory
    directory = "." if getattr(cls, directory_key, None) is None else getattr(cls, directory_key)
    directory = os.path.abspath(directory)
    # if we've overloaded any excluded key, aren't using the npz file yet, or are changing directories (re)make npz
    make_npz = False
    for key in excluded_keys:
        make_npz = make_npz or isinstance(getattr(cls, key), np.ndarray)
    if isinstance(getattr(cls, npz_key), str):
        file_name = getattr(cls, npz_key)
        if os.path.dirname(file_name) != directory:
            make_npz = True
    if make_npz:
        # get the data to save
        excluded_data = {}
        for key in excluded_keys:
            if getattr(cls, key) is not None:
                excluded_data[key] = getattr(cls, key)
        # if there is data to save, save it
        if excluded_data:
            # get the npz file name
            file_name = tempfile.mkstemp(prefix=prefix, suffix=".npz", dir=directory)[1]
            np.savez(file_name, **excluded_data)
            setattr(cls, npz_key, file_name)
    # change the excluded keys in the dict to the key for the npz_file if it exists
    if getattr(cls, npz_key) is not None:
        for key in excluded_keys:
            cls.__dict__[key] = key
    return cls.__dict__


def quadratic_spline_roots(spline):
    """Returns the roots of a scipy spline."""
    roots = []
    knots = spline.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spline(a), spline((a + b) / 2), spline(b)
        t = np.roots([u + w - 2 * v, w - u, 2 * v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t * (b - a) / 2 + (b + a) / 2)
    return np.array(roots)


def ev_nm_convert(x):
    """
    If x is a wavelength in nm, the corresponding energy in eV is returned.
    If x is an energy in eV, the corresponding wavelength in nm is returned.
    """
    return c.speed_of_light * c.h / c.eV * 1e9 / x


def load_legacy_binary_data(binary_file, channel, n_points, noise=True):
    """
    Load data from legacy Matlab code binary files (.ns or .dat).
    Args:
        binary_file: string
            The full file name and path to the binary data.
        channel: integer
            Either 0 or 1 specifying which channel to load data from.
        n_points: integer
            The number of points per trigger per trace. Both I and Q traces
            should have the same number of points.
        noise: boolean (optional)
            A flag specifying if the data is in the noise or pulse format.
    Returns:
        i_trace: numpy.ndarray
            An N x n_points numpy array with N traces.
        q_trace: numpy.ndarray
            An N x n_points numpy array with N traces.
        f: float
            The bias frequency for the noise data.
    """
    # get the binary data from the file
    data = np.fromfile(binary_file, dtype=np.int16)
    # grab the tone frequency (not a 16 bit integer)
    if noise:
        f = np.frombuffer(data[4 * channel: 4 * (channel + 1)].tobytes(), dtype=np.float64)[0]
    else:
        f = np.frombuffer(data[4 * (channel + 2): 4 * (channel + 3)].tobytes(), dtype=np.float64)[0]
    # remove the header from the file
    data = data[4 * 12:] if noise else data[4 * 14:]
    # convert the data to voltages * 0.2 V / (2**15 - 1)
    data = data.astype(np.float16) * 0.2 / 32767.0
    # check that we have an integer number of triggers
    n_triggers = data.size / n_points / 4.0
    assert n_triggers.is_integer(), "non-integer number of noise traces found found in {0}".format(binary_file)
    n_triggers = int(n_triggers)
    # break noise data into I and Q data
    i_trace = np.zeros((n_triggers, n_points), dtype=np.float16)
    q_trace = np.zeros((n_triggers, n_points), dtype=np.float16)
    for trigger_num in range(n_triggers):
        trace_num = 4 * trigger_num
        i_trace[trigger_num, :] = data[(trace_num + 2 * channel) * n_points: (trace_num + 2 * channel + 1) * n_points]
        q_trace[trigger_num, :] = data[(trace_num + 2 * channel + 1) * n_points:
                                       (trace_num + 2 * channel + 2) * n_points]
    return i_trace, q_trace, f


def structured_to_complex(array):
    if array is None or array.dtype == np.complex or array.dtype == np.complex64:
        return array
    else:
        return array["I"] + 1j * array["Q"]


def lmfit(lmfit_results, model, guess, label='default', residual_args=(), residual_kwargs=None, model_index=None,
          **kwargs):
    if label == 'best':
        raise ValueError("'best' is a reserved label and cannot be used")
    # set up and do minimization
    minimizer = lm.Minimizer(model.residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
    result = minimizer.minimize(**kwargs)
    # save the results
    if model_index is not None:
        model = model.models[model_index]  # if the fit was done with a joint model only save the relevant part
        residual_args = tuple([arg[model_index] if isinstance(arg, tuple) else arg for arg in residual_args])
        residual_kwargs = {key: value[model_index] if isinstance(value, tuple) else value
                           for key, value in residual_kwargs.items()}
    save_lmfit(lmfit_results, model, result, label=label, residual_args=residual_args, residual_kwargs=residual_kwargs)


def save_lmfit(lmfit_results, model, result, label='default', residual_args=(), residual_kwargs=None):
    if label in lmfit_results:
        log.warning("'{}' has already been used as an lmfit label. The old data has been overwritten.".format(label))
    lmfit_results[label] = {'result': result, 'model': model, 'kwargs': residual_kwargs, 'args': residual_args}
    # if the result is better than has been previously computed, add it to the 'best' key
    if 'best' not in lmfit_results.keys():
        lmfit_results['best'] = lmfit_results[label]
        lmfit_results['best']['label'] = label
    elif result.aic < lmfit_results['best']['result'].aic:
        lmfit_results['best'] = lmfit_results[label]
        lmfit_results['best']['label'] = label


def create_range(value):
    if value is None:
        value = ((-np.inf, np.inf),)
    elif not isinstance(value, (tuple, list, np.ndarray)):
        value = ((value, value),)
    elif len(value) == 2 and (isinstance(value[0], numbers.Number) and isinstance(value[1], numbers.Number)):
        value = (value,)
    return value


def create_ranges(power, field, temperature):
    power = create_range(power)
    field = create_range(field)
    temperature = create_range(temperature)
    return power, field, temperature


def sort_and_fix(data, energies, fix_zero):
    data = [0] + data if fix_zero else phase
    data, energies = np.array(data), np.array(energies)
    energies, indices = np.unique(energies, return_index=True)
    data = data[indices]
    return data, energies


def setup_axes(axes, x_label, y_label, label_kwargs=None, x_label_default="", y_label_default="", equal=False):
    if axes is None:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure
    if x_label is None:
        x_label = x_label_default
    if y_label is None:
        y_label = y_label_default
    # setup axes
    kwargs = {}
    if label_kwargs is not None:
        kwargs.update(label_kwargs)
    if x_label:
        axes.set_xlabel(x_label, **kwargs)
    if y_label:
        axes.set_ylabel(y_label, **kwargs)
    if equal:
        axes.axis('equal')
    return figure, axes


def finalize_axes(axes, title=False, title_kwargs=None, legend=False, legend_kwargs=None, tick_kwargs=None,
                  tighten=False):
    if legend:
        kwargs = {}
        if legend_kwargs is not None:
            kwargs.update(legend_kwargs)
        axes.legend(**kwargs)
    # make the title
    if title:
        kwargs = {"fontsize": 11}
        if title_kwargs is not None:
            kwargs.update(title_kwargs)
        axes.set_title(text, **kwargs)
    if tick_kwargs is not None:
        axes.tick_params(**tick_kwargs)
    if tighten:
        axes.figure.tight_layout()


def get_plot_model(self, fit_type, label, params=None, calibrate=False, default_kwargs=None, plot_kwargs=None,
                   center=False, n_factor=10):
    # get the model
    fit_name, result_dict = self._get_model(fit_type, label)
    if fit_name is None:
        raise ValueError("No fit of type '{}' with the label '{}' has been done".format(fit_type, label))
    model = result_dict['model']
    # calculate the model values
    f = np.linspace(np.min(self.f), np.max(self.f), np.size(self.f) * n_factor)
    if params is None:
        params = result_dict['result'].params
    m = model.model(params, f)
    if calibrate:
        m = model.calibrate(params, m, f, center=center)
    # add the plot
    kwargs = {} if default_kwargs is None else default_kwargs
    if plot_kwargs is not None:
        kwargs.update(plot_kwargs)
    return f, m, kwargs


def _integer_bandwidth(f, df):
    return int(np.round(df / (f[1] - f[0]) / 2) * 2)  # nearest even number


def find_resonators(f, z, df, **kwargs):
    """
    Find resonators in a S21 trace.
    Args:
        f: numpy.ndarray
            The frequencies corresponding to the magnitude array.
        z: numpy.ndarray
            The complex scattering data.
        df: float
            The frequency bandwidth for each resonator. df / 2 will be used as
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
    magnitude = detrend(20 * np.log10(np.abs(z)))
    fit = np.argsort(magnitude)[:int(3 * len(magnitude) / 4):-1]
    poly = np.polyfit(f[fit], magnitude[fit], 1)
    magnitude = magnitude - np.polyval(poly, f)
    # find peaks
    kws = {"prominence": 1, "height": 5, "width": (None, int(dfii / 2))}
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
    Collect all of the resonances from a data into an array.
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
    f_array = np.empty((len(peaks), int(dfii)))
    z_array = np.empty(f_array.shape, dtype=np.complex)
    for ii in range(f_array.shape[0]):
        f_array[ii, :] = f[int(peaks[ii] - dfii / 2): int(peaks[ii] + dfii / 2)]
        z_array[ii, :] = z[int(peaks[ii] - dfii / 2): int(peaks[ii] + dfii / 2)]
    # cut out resonators that aren't centered (large resonator tails on either side)
    logic = np.abs(np.argmin(np.abs(z_array), axis=-1) - dfii / 2) > dfii / 10
    return f_array[logic, :], z_array[logic, :], peaks[logic]


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
                        tighten=True, scatter=True, median=True, bins=30, extend=True, return_bin=False, axes=None,
                        median_kwargs=None, scatter_kwargs=None):
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
        scatter: boolean (optional)
            Whether to plot a scatter of the data as a function of frequency.
        median: boolean (optional)
            Whether to plot the median as a function of frequency.
        bins: integer (optional)
            The number of bins to use in the median plot. The default is 30.
        extend: boolean (optional)
            Determines whether or not to extend the median data so that there
            is a bin with zero values on either side of the frequency range.
            The default is True.
        return_bin: boolean (optional)
            Whether or not to include the binned median information in the
            returned values. The default is False and only the axes are
            returned. The bin values are not returned if median is False.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
        median_kwargs: dictionary (optional)
            Extra keyword arguments to send to axes.step().
        scatter_kwargs: dictionary (optional)
            Extra keyword arguments to send to axes.plot().
        Returns:
            axes: matplotlib.axes.Axes class
                An Axes class with the plotted data.
            centers: numpy.ndarray
                The bin centers. Only returned if return_bin and median are
                True.
            medians: numpy.ndarray
                The median values in each bin. Only returned if return_bin and
                median are True.
    """
    if axes is None:
        from matplotlib import pyplot as plt
        figure, axes = plt.subplots()
    else:
        figure = axes.figure

    if scatter:
        kws = {"linestyle": "none"}
        if scatter_kwargs is not None:
            kws.update(scatter_kwargs)
        axes.plot(f, parameter)
    if median:
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
        kws = {"where": "mid"}
        if median_kwargs is not None:
            kws.update(median_kwargs)
        axes.step(centers, medians, **kws)
    else:
        centers = None
        medians = None
    axes.set_xlim(f.min(), f.max())
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
    if return_bin and median:
        return axes, centers, medians
    return axes
