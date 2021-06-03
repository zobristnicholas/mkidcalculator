import calendar
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from mkidcalculator.io.utils import setup_axes, finalize_axes


def load_lakeshore_log(filename, start=None, stop=None, timestamp=False,
                       absolute=True):
    """
    Load the log file and return the times and values.
    Args:
        filename: string
            The log file name including the path.
        start: datetime.datetime
            The start time for the experiment. If not provided, the whole
            log file is used.
        stop: datetime.datetime
            The stop time for the experiment. If not provided, the whole
            log file is used.
        timestamp: boolean
            If True, the dates are returned as timestamps. The default is
            False and datetime objects are returned.
        absolute: boolean
            The default is True and the absolute value of the data is
            returned. If False, the original data is returned instead.
    Returns:
        time: numpy.ndarray
            Times corresponding to the values.
        values: numpy.ndarray
            Values at the corresponding times.
    """
    # Load the file.
    data = np.loadtxt(filename, dtype=np.object, delimiter=",")
    # Reformat the time into a datetime.
    time_strings = [t[0].strip() + " " + t[1].strip() for t in data]
    fmt = "%d-%m-%y %H:%M:%S"
    time = np.array([datetime.strptime(t, fmt) for t in time_strings])
    # Coerce the values into floats.
    values = data[:, 2].astype(float)
    # Mask the data.
    mask = np.ones(time.shape, dtype=bool)
    if start is not None:
        mask *= (time >= start)
    if stop is not None:
        mask *= (time <= stop)
    time = time[mask]
    values = values[mask]
    # Convert times to timestamps for easy manipulation.
    if timestamp:
        time = np.array([t.timestamp() for t in time])
    # Return absolute values.
    if absolute:
        values = np.abs(values)
    return time, values


def t_vs_r(t_filename, r_filename, start=None, stop=None):
    """
    Return the resistance as a function of temperature.
    Args:
        t_filename: string
            The filename for the temperature log data.
        r_filename: string
            The file name for the resistance log data.
        start: datetime.datetime
            The start time for the experiment. If not provided, the whole
            log file is used.
        stop: datetime.datetime
            The stop time for the experiment. If not provided, the whole
            log file is used.
    Returns:
        temperature: numpy.ndarray
            Temperature at the same time as the resistance.
        resistance: numpy.ndarray
            Resistance at the same time as the temperature.
    """
    # Load the files.
    time_t, t = load_lakeshore_log(t_filename, start=start, stop=stop,
                                   timestamp=True)
    time_r, resistance = load_lakeshore_log(r_filename, start=start,
                                            stop=stop, timestamp=True)
    # Get the temperature when the resistance was measured by interpolation.
    time_to_t = interp1d(time_t, t, fill_value='extrapolate')
    temperature = time_to_t(time_r)
    return temperature, resistance


def plot_transition(t_filename, r_filename, start=None, stop=None,
                    plot_kwargs=None, x_label=None, y_label=None,
                    label_kwargs=None, legend=False, legend_kwargs=None,
                    title=False, title_kwargs=None, tick_kwargs=None,
                    tighten=True, axes=None):
    """
    Plot the transition data.
    Args:
        t_filename: string
            The filename for the temperature log data.
        r_filename: string
            The file name for the resistance log data.
        start: datetime.datetime
            The start time for the experiment. If not provided, the whole
            log file is used.
        stop: datetime.datetime
            The stop time for the experiment. If not provided, the whole
            log file is used.
        x_label: string
            The label for the x axis. The default is None which uses the
            default label. If x_label evaluates to False, parameter_kwargs
            is ignored.
        y_label: string
            The label for the y axis. The default is None which uses the
            default label. If y_label evaluates to False, parameter_kwargs
            is ignored.
        plot_kwargs: dictionary
            Keyword arguments for the data in axes.plot(). The default is
            None which uses default options. Keywords in this dictionary
            override the default options.
        label_kwargs: dictionary
            Keyword arguments for the axes labels in axes.set_*label(). The
            default is None which uses default options. Keywords in this
            dictionary override the default options.
        legend: boolean
            Determines whether the legend is used or not. The default is
            True. If False, legend_kwargs is ignored.
        legend_kwargs: dictionary
            Keyword arguments for the legend in axes.legend(). The default
            is None which uses default options. Keywords in this
            dictionary override the default options.
        title: boolean or string
            If it is a boolean, it determines whether or not to add the
            default title. If it is a string, that string is used as the
            title. If False, title_kwargs is ignored. The default is False.
        title_kwargs: dictionary
            Keyword arguments for the axes title in axes.set_title(). The
            default is None which uses default options. Keywords in this
            dictionary override the default options.
        tick_kwargs: dictionary
            Keyword arguments for the ticks using axes.tick_params(). The
            default is None which uses the default options. Keywords in
            this dictionary override the default options.
        tighten: boolean
            Determines whether figure.tight_layout() is called. The default
            is True.
        axes: matplotlib.axes.Axes class
            An Axes class on which to put the plot. The default is None and
            a new figure is made.
    Returns:
        axes: matplotlib.axes.Axes class
            An Axes class with the plotted loop.
    """
    t, r = t_vs_r(t_filename, r_filename, start=start, stop=stop)
    _, axes = setup_axes(axes, x_label, y_label, label_kwargs,
                         'T [mK]', r'R [$\Omega$]')
    kwargs = {"marker": 'o', "markersize": 4, "linestyle": "none",
              "markerfacecolor": "k", "markeredgecolor": "none"}
    if plot_kwargs is not None:
        kwargs.update(plot_kwargs)
    axes.plot(t * 1e3, r, **kwargs)
    title = "superconducting transition" if title is True else title
    finalize_axes(axes, title=title, title_kwargs=title_kwargs, legend=legend,
                  legend_kwargs=legend_kwargs, tick_kwargs=tick_kwargs,
                  tighten=tighten)
    return axes
