import os
import pickle
import matplotlib
import numpy as np
import pandas as pd
from operator import itemgetter
from scipy.cluster.vq import kmeans2, ClusterError

from mkidcalculator.io.loop import Loop
from mkidcalculator.io.data import analogreadout_sweep


class Sweep:
    def __init__(self):
        self.loops = []
        self.powers = []
        self.fields = []
        self.temperatures = []
        self.temperature_groups = []
        # analysis results
        self.loop_parameters = {}
        # directory of the data
        self._directory = None

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Sweep class from the pickle file 'file_name'."""
        with open(file_name, "rb") as f:
            sweep = pickle.load(f)
        assert isinstance(sweep, cls), "'{}' does not contain a Sweep class.".format(file_name)
        return sweep

    def create_parameters(self, label="best", fit_type="lmfit", group=True, n_groups=None):
        """
        Creates the loop parameters pandas DataFrame by looking at all of the
        loop fits. in
        Args:
            label: string
                Corresponds to the label in the loop.lmfit_results or
                loop.emcee_results dictionaries where the fit parameters are.
                The resulting DataFrame is stored in
                self.loop_parameters[label]. The default is "best", which gets
                the parameters from the best fits.
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            group: boolean
                Determines if the temperature data is grouped together in the
                table. This is useful for when data is taken at the same
                temperature but the actual temperature has some fluctuations.
                The default is True, and the actual temperature is stored under
                the column 'temperature'. If False, n_groups is ignored.
            n_groups: integer
                An integer that determines how many temperature groups to
                include. The default is None, and n_groups is calculated. This
                procedure only works if the data is 'square' (same number of
                temperature points per unique power and field combination).
        Raises:
            scipy.cluster.vq.ClusterError:
                The temperature data is too disordered to cluster into the
                specified number of groups.
        """
        # check inputs
        if fit_type not in ['lmfit', 'emcee', 'emcee_mle']:
            raise ValueError("'fit_type' must be either 'lmfit', 'emcee', or 'emcee_mle'")
        # group temperatures
        if group:
            temperatures = np.array(self.temperatures)
            if n_groups is None:
                n_groups = temperatures.size // (np.unique(self.powers).size * np.unique(self.fields).size)
            k = np.linspace(temperatures.min(), temperatures.max(), n_groups)
            try:
                centroids, groups = kmeans2(temperatures, k=k, minit='matrix', missing='raise')
            except ClusterError:
                message = "The temperature data is too disordered to cluster into {} groups".format(n_groups)
                raise ClusterError(message)
            self.temperature_groups = np.empty(temperatures.shape)
            for index, centroid in enumerate(centroids):
                self.temperature_groups[groups == index] = centroid
            self.temperature_groups = list(self.temperature_groups)
        else:
            self.temperature_groups = self.temperatures
        # determine the parameter names
        parameter_names = set()
        parameters = []
        for loop in self.loops:
            _, result_dict = loop._get_model(fit_type, label)
            p = result_dict['result'].params.valuesdict() if result_dict is not None else None
            # save the parameters and collect all the names into the set
            parameters.append(p)
            if p is not None:
                for name in p.keys():
                    parameter_names.add(name)
        # initialize the data frame
        parameter_names = sorted(list(parameter_names))
        if group:
            if "temperature" in parameter_names:
                raise ValueError("'temperature' can not be a fit parameter name if group=True")
            parameter_names.append("temperature")
        indices = list(zip(self.powers, self.fields, self.temperature_groups))
        multi_index = pd.MultiIndex.from_tuples(indices, names=["power", "field", "temperature"])
        df = pd.DataFrame(np.nan, index=multi_index, columns=parameter_names)
        # fill the data frame
        for index, loop in enumerate(self.loops):
            if parameters[index] is not None:
                for key, value in parameters[index].items():
                    df.loc[indices[index]][key] = value
            if group:
                df.loc[indices[index]]["temperature"] = self.temperatures[index]
        self.loop_parameters[label] = df

    def add_loops(self, loops, sort=True):
        """
        Add loop data sets to the sweep.
        Args:
            loops: Loop class or iterable of Loop classes
                The loop data sets that are to be added to the Sweep.
            sort: boolean (optional)
                Sort the loop data list by its power, field, and temperature.
                The default is True. If False, the order of the loop data sets
                is preserved.
        """
        if isinstance(loops, Loop):
            loops = [loops]
        # append loop data
        for loop in loops:
            self.loops.append(loop)
            self.powers.append(loop.power)
            self.fields.append(loop.field)
            self.temperatures.append(loop.temperature)
        # sort
        if sort:
            lp = zip(*sorted(zip(self.powers, self.fields, self.temperatures, self.loops), key=itemgetter(0, 1, 2)))
            self.powers, self.fields, self.temperatures, self.loops = (list(t) for t in lp)

    def remove_loops(self, indices):
        """
        Remove loops from the sweep.
        Args:
            indices: integer or iterable of integers
                The indices in sweep.loops that should be deleted.
        """
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        for ii in sorted(indices, reverse=True):
            self.loops.pop(ii)
            self.powers.pop(ii)
            self.fields.pop(ii)
            self.temperatures.pop(ii)

    def free_memory(self, directory=None):
        """
        Frees memory from all of the contained Loop objects.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the pulse was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        for loop in self.loops:
            loop.free_memory()
        _loaded_npz_files.free_memory(self._data._npz)

    @classmethod
    def from_config(cls, sweep_file_name, data=analogreadout_sweep, sort=True, **kwargs):
        """
        Sweep class factory method that returns a Sweep() with the loop, noise
        and pulse data loaded.
        Args:
            sweep_file_name: string
                The configuration file name for the sweep data. It has all the
                information needed to load in the loop, pulse, and noise data.
            data: object (optional)
                Class or function whose return value is a list of dictionaries
                with each being the desired keyword arguments to Loop.load().
            sort: boolean (optional)
                Sort the loop data by its power, field, and temperature. Also
                sort noise data and pulse data lists by their bias frequency.
                The default is True. If False, the input order is preserved.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to Loop.load() not
                specified by data.
        """
        # create sweep
        sweep = cls()
        # load loop kwargs based on the sweep file
        loop_kwargs_list = data(sweep_file_name)
        loops = []
        # load loops
        for kws in loop_kwargs_list:
            kws.update(kwargs)
            kws.update({"sort": sort})
            loops.append(Loop.load(**kws))
        sweep.add_loops(loops, sort=sort)
        return sweep

    def lmfit(self):
        raise NotImplementedError

    def emcee(self):
        raise NotImplementedError

    def _set_directory(self, directory):
        self._directory = directory
        for loop in self.loops:
            loop._set_directory(self._directory)

    def plot_loops(self, power=None, field=None, temperature=None, color_data='temperature', colormap=None,
                   colorbar=True, colorbar_kwargs=None, colorbar_label=True, colorbar_label_kwargs=None, **loop_kwargs):
        """
        Plot a subset of the loops in the sweep by combining multiple
        loop.plot() calls.
        Args:
            power: number or tuple of two numbers
                Inclusive range of powers to plot. A single number will cause
                only that value to be plotted. The default is to include all of
                the powers.
            field: number or tuple of two numbers
                Inclusive range of fields to plot. A single number will cause
                only that value to be plotted. The default is to include all of
                the fields.
            temperature: number or tuple of two numbers
                Inclusive range of temperatures to plot. A single number will
                cause only that value to be plotted. The default is to include
                all of the temperatures.
            color_data: string
                Either 'temperature', 'field', or 'power' indicating off what
                type of data to base the colormap. The default is
                'temperature'.
            colormap: matplotlib.colors.Colormap
                A matplotlib colormap for coloring the data. If the default
                None is used, a colormap is chosen based on color_data.
            colorbar: boolean
                Determines whether to include a colorbar. The default is True.
                If False, colorbar_kwargs, colorbar_label, and
                colorbar_label_kwargs are ignored.
            colorbar_kwargs: dictionary
                Keyword arguments for the colorbar in figure.colorbar(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            colorbar_label: boolean or string
                If it is a boolean, it determines whether or not to add the
                default colorbar label. If it is a string, that string is used
                as the colorbar label. If False, colorbar_label_kwargs is
                ignored. The default is True.
            colorbar_label_kwargs: dictionary
                Keyword arguments for the colorbar in colorbar.set_label(). The
                default is None which uses default options. Keywords in this
                dictionary override the default options.
            loop_kwargs: optional keyword arguments
                Extra keyword arguments to send to loop.plot().
        Returns:
            axes_list: an iterable of matplotlib.axes.Axes classes
                A list of Axes classes with the plotted data.
        """
        # parse inputs
        if "fit_parameters" in loop_kwargs.keys():
            raise TypeError("'fit_parameters' is not a valid keyword argument")
        if "parameters_kwargs" in loop_kwargs.keys():
            raise TypeError("'parameters_kwargs' is not a valid keyword argument")
        if power is None:
            power = (-np.inf, np.inf)
        elif not isinstance(power, (tuple, list, np.ndarray)):
            power = (power, power)
        if field is None:
            field = (-np.inf, np.inf)
        elif not isinstance(field, (tuple, list, np.ndarray)):
            field = (field, field)
        if temperature is None:
            temperature = (-np.inf, np.inf)
        elif not isinstance(temperature, (tuple, list, np.ndarray)):
            temperature = (temperature, temperature)
        if color_data == 'temperature':
            cmap = matplotlib.cm.get_cmap('coolwarm') if colormap is None else colormap
            cdata = np.array(self.temperatures) * 1000
        elif color_data == 'field':
            cmap = matplotlib.cm.get_cmap('viridis') if colormap is None else colormap
            cdata = self.fields
        elif color_data == 'power':
            cmap = matplotlib.cm.get_cmap('plasma') if colormap is None else colormap
            cdata = self.powers
        else:
            raise ValueError("'{}' is not a valid value of color_data.".format(color_data))
        norm = matplotlib.colors.Normalize(vmin=min(cdata), vmax=max(cdata))
        n_plots = 3 if 'plot_types' not in loop_kwargs.keys() else len(loop_kwargs['plot_types'])
        axes_list = None
        # format title
        title = loop_kwargs.get("title", True)
        if title is True:
            # power
            if power[0] == power[1]:
                title = "{:.0f} dBm, ".format(power[0])
            elif np.isinf(power[0]) and np.isinf(power[1]):
                title = "All Powers, "
            else:
                title = "({:.0f}, {:.0f}) dBm, ".format(power[0], power[1])
            # field
            if field[0] == field[1]:
                title += "{:.0f} V, ".format(field[0])
            elif np.isinf(field[0]) and np.isinf(field[1]):
                title += "All Fields, "
            else:
                title += "({:.0f}, {:.0f}) V, ".format(field[0], field[1])
            # temperature
            if temperature[0] == temperature[1]:
                title += "{:.0f} mK".format(temperature[0] * 1000)
            elif np.isinf(temperature[0]) and np.isinf(temperature[1]):
                title += "All Temperatures"
            else:
                title += "({:.0f}, {:.0f}) mK".format(temperature[0] * 1000, temperature[1] * 1000)
        # store key word options
        user_plot_kwargs = loop_kwargs.get('plot_kwargs', [])
        user_data_kwargs = []
        for kw in user_plot_kwargs:
            user_data_kwargs.append(kw.get("data_kwargs", {}))
        user_fit_kwargs = []
        for kw in user_plot_kwargs:
            user_fit_kwargs.append(kw.get("fit_kwargs", {}))
        # make a plot for each loop
        plot_index = 0
        for index, loop in enumerate(self.loops):
            condition = (power[0] <= loop.power <= power[1] and
                         field[0] <= loop.field <= field[1] and
                         temperature[0] <= loop.temperature <= temperature[1])
            if condition:
                # default plot key words
                if plot_index == 0:
                    plot_kwargs = [{'data_kwargs': {'color': cmap(norm(cdata[index]))},
                                    'fit_kwargs': {'color': 'k'}}] * n_plots
                else:
                    plot_kwargs = [{'x_label': '', 'y_label': '', 'data_kwargs': {'color': cmap(norm(cdata[index]))},
                                    'fit_kwargs': {'color': 'k'}}] * n_plots
                # update data key words with user defaults
                for kw_index, data_kw in enumerate(user_data_kwargs):
                    plot_kwargs[kw_index]['data_kwargs'].update(data_kw)
                # update fit key words with user defaults
                for kw_index, fit_kw in enumerate(user_fit_kwargs):
                    plot_kwargs[kw_index]['fit_kwargs'].update(fit_kw)
                # update plot key words with user defaults
                for kw_index, kws in enumerate(user_plot_kwargs):
                    kws = kws.copy()
                    kws.pop('data_kwargs')
                    kws.pop('fit_kwargs')
                    plot_kwargs[kw_index].update(kws)
                # update loop kwargs
                if plot_index == 0:
                    loop_kwargs.update({"plot_kwargs": plot_kwargs, "title": title})
                else:
                    loop_kwargs.update({"axes_list": axes_list, "title": False, "legend": False, "tighten": False,
                                        "plot_kwargs": plot_kwargs})
                axes_list = loop.plot(**loop_kwargs)
                plot_index += 1
        # if we didn't plot anything exit the function
        if axes_list is None:
            return
        if colorbar:
            mappable = matplotlib.cm.ScalarMappable(norm, cmap)
            mappable.set_array([])
            kwargs = {'aspect': 30}
            if colorbar_kwargs is not None:
                kwargs.update(colorbar_kwargs)
            cbar = axes_list[0].figure.colorbar(mappable, ax=axes_list, **kwargs)
            if colorbar_label:
                if color_data == 'temperature':
                    label = "temperature [mK]" if colorbar_label is True else colorbar_label
                elif color_data == 'field':
                    label = "field [V]" if colorbar_label is True else colorbar_label
                else:
                    label = "power [dBm]" if colorbar_label is True else colorbar_label
                kwargs = {"rotation": 270, 'va': 'bottom'}
                if colorbar_label_kwargs is not None:
                    kwargs.update(colorbar_label_kwargs)
                cbar.set_label(label, **kwargs)
            cbar_width = cbar.ax.get_window_extent().transformed(axes_list[0].figure.dpi_scale_trans.inverted()).width
            axes_list[0].figure.set_figwidth(axes_list[0].figure.get_figwidth() + cbar_width)
        return axes_list
