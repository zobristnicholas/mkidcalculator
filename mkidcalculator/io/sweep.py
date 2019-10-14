import os
import pickle
import logging

from mkidcalculator.io.loop import Loop
from mkidcalculator.io.resonator import Resonator
from mkidcalculator.io.data import analogreadout_sweep, mazinlab_widesweep
from mkidcalculator.plotting import plot_parameter_vs_f, plot_parameter_hist
from mkidcalculator.io.utils import find_resonators, collect_resonances, _loop_fit_data

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Sweep:
    """A class for organizing data from multiple resonators."""
    def __init__(self):
        self.resonators = []

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        with open(file_name, "wb") as f:
            pickle.dump(self, f)
        log.info("saved sweep as '{}'".format(file_name))

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Sweep class from the pickle file 'file_name'."""
        with open(file_name, "rb") as f:
            sweep = pickle.load(f)
        assert isinstance(sweep, cls), "'{}' does not contain a Sweep class.".format(file_name)
        log.info("loaded sweep from '{}'".format(file_name))
        return sweep

    def add_resonators(self, resonators):
        """
        Add Resonator objects to the sweep.
        Args:
            resonators: Resonator class or iterable of Resonator classes
                The resonators that are to be added to the Sweep.
        """
        if isinstance(resonators, Resonator):
            resonators = [resonators]
        # append resonator data
        for resonator in resonators:
            resonator.sweep = self
            self.resonators.append(resonator)

    def remove_resonators(self, indices):
        """
        Remove resonators from the sweep.
        Args:
            indices: integer or iterable of integers
                The indices in sweep.resonators that should be deleted.
        """
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        for ii in sorted(indices, reverse=True):
            self.resonators.pop(ii)

    def free_memory(self, directory=None):
        """
        Frees memory from all of the contained Resonator objects.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the pulse was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        for resonator in self.resonators:
            resonator.free_memory(directory=directory)

    @classmethod
    def from_widesweep(cls, sweep_file_name, df, data=mazinlab_widesweep, find=find_resonators, find_kwargs=None,
                       loop_kwargs=None, **kwargs):
        """
        Sweep class factory method that returns a Sweep() from widesweep data
        the resonators identified and loaded.
        Args:
            sweep_file_name: string
                The file name for the widesweep data.
            df: float
                The frequency bandwidth for each resonator in the units of the
                data in the file.
            data: object (optional)
                Function whose return value is a tuple of the frequencies
                (numpy.ndarray), complex scattering data (numpy.ndarray),
                attenuation (float), field (float), and temperature (float) of
                the widesweep.
            find: object (optional)
                Function whose return value is a list of resonator peak indices
                corresponding to the data returned by 'data'. The manditory
                input arguments are f, z, and df.
            find_kwargs: dictionary (optional)
                Extra keyword arguments to pass to the find function.
            loop_kwargs: dictionary (optional)
                Extra keyword arguments to pass to loop.from_python().
            kwargs: optional keyword arguments
                Extra keyword arguments to send to data.
        Returns:
            sweep: object
                A Sweep() object containing the loaded data.
        """
        sweep = cls()
        f, z, attenuation, field, temperature = data(sweep_file_name, **kwargs)
        kws = {}
        if find_kwargs is not None:
            kws.update(find_kwargs)
        peaks = find(f, z, df, **kws)
        f_array, z_array, _ = collect_resonances(f, z, peaks, df)
        resonators = []
        for ii in range(f_array.shape[0]):
            zii, fii = z_array[ii, :], f_array[ii, :]
            resonators.append(Resonator())
            kws = {}
            if loop_kwargs is not None:
                kws.update(loop_kwargs)
            resonators[-1].add_loops(Loop.from_python(zii, fii, attenuation, field, temperature, **kws))
        sweep.add_resonators(resonators)
        return sweep

    @classmethod
    def from_file(cls, sweep_file_name, data=analogreadout_sweep, sort=True, **kwargs):
        """
        Sweep class factory method that returns a Sweep() with the resonator
        data loaded.
        Args:
            sweep_file_name: string
                The file name for the sweep data.
            data: object (optional)
                Class or function whose return value is a list of dictionaries
                with each being the desired keyword arguments to
                Resonator.from_file().
            sort: boolean (optional)
                Sort the loop data in each resonator by its power, field, and
                temperature. Also sort noise data and pulse data lists for each
                loop by their bias frequencies. The default is True. If False,
                the input order is preserved. The resonators list is not
                sorted.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to data.
        Returns:
            sweep: object
                A Sweep() object containing the loaded data.
        """
        sweep = cls()
        res_kwarg_list = data(sweep_file_name, **kwargs)
        resonators = []
        for kws in res_kwarg_list:
            kws.update({"sort": sort})
            resonators.append(Resonator.from_file(**kws))
        sweep.add_resonators(resonators)
        return sweep

    def _set_directory(self, directory):
        self._directory = directory
        for resonator in self.resonators:
            resonator._set_directory(self._directory)

    def plot_loop_fits(self, parameters=("qi", "q0", "chi2"), bounds=None, errorbars=True, success=True,
                       title=True, tighten=True, label='best', plot_kwargs=None, figure=None):
        """
        Plot a summary of all the loop fits.
        Args:
            parameters: tuple of strings
                The fit parameters to plot. "chi2" can be used to plot the
                reduced chi squared value.
            bounds: tuple of numbers or tuples
                The bounds for the parameters. It must be a tuple of the same
                length as the parameters keyword argument. Each element is either
                an upper bound on the parameter or a tuple, e.g. (lower bound,
                upper bound). None can be used as a placeholder to skip a
                bound. The default is None and no bounds are used.
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
        loops = []
        for resonator in self.resonators:
            loops += resonator.loops
        parameters = ["fr"] + list(parameters)
        outputs = _loop_fit_data(loops, parameters=parameters, label=label, bounds=bounds, success=success,
                                 errorbars=errorbars)
        # create figure if needed
        if figure is None:
            from matplotlib import pyplot as plt
            figure = plt.figure(figsize=(8.5, 11))
        # setup figure axes
        gs = gridspec.GridSpec(len(parameters) - 1, 2)
        axes_list = np.array([figure.add_subplot(gs_ii) for _, gs_ii in np.ndenumerate(gs)])
        # check plot kwargs
        if plot_kwargs is None:
            plot_kwargs = {}
        if isinstance(plot_kwargs, dict):
            plot_kwargs = [plot_kwargs] * len(axes_list)
        # add plots
        for index in range(axes_list // 2):
            kws = {"x_label": parameters[index + 1]}
            if plot_kwargs[0]:
                kws.update(plot_kwargs[2 * index])
            plot_parameter_hist(outputs[index + 1], axes=axes_list[0], **kws)
            kws = {"y_label": "median " + parameters[index + 1]}
            if plot_kwargs[1]:
                kws.update(plot_kwargs[2 * index + 1])
            plot_parameter_vs_f(outputs[index + 1], outputs[0], axes=axes_list[1], **kws)
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
