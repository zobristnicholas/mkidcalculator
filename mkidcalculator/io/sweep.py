import os
import pickle
import logging
import inspect
import matplotlib
import lmfit as lm
import numpy as np
import pandas as pd
from operator import itemgetter
from scipy.cluster.vq import kmeans2, ClusterError

from mkidcalculator.io.loop import Loop
from mkidcalculator.io.data import analogreadout_sweep
from mkidcalculator.io.utils import lmfit, create_ranges, save_lmfit

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Sweep:
    """A class for manipulating resonance sweep parameter data."""
    def __init__(self):
        self.loops = []
        self.powers = []
        self.fields = []
        self.temperatures = []
        self.temperature_groups = []
        # analysis results
        self.lmfit_results = {}
        self.loop_parameters = {}
        # directory of the data
        self._directory = None

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

    def create_parameters(self, label="best", fit_type="lmfit", group=True, n_groups=None):
        """
        Creates the loop parameters pandas DataFrame by looking at all of the
        loop fits.
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

        Examples:
            table = sweep.loop_parameters['best']
            # get a table with only the 'fr' fit parameter
            fr = table['fr']
            # get a smaller table with all of the powers, zero field, and
            # temperatures between 9 and 11 mK
            idx = pandas.IndexSlice
            fr_smaller = fr.loc[idx[:, 0, 0.009: 0.011]]
            # get a cross section instead of a table
            fr_smaller = fr.xs((idx[:], 0,  idx[0.009:0.011]), level=("power", "field", "temperature"))
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
        results = []
        for loop in self.loops:
            _, result_dict = loop._get_model(fit_type, label)
            results.append(result_dict['result'])
            p = result_dict['result'].params if result_dict is not None else None
            # save the parameters and collect all the names into the set
            parameters.append(p)
            if p is not None:
                for name in p.keys():
                    parameter_names.add(name)
                    parameter_names.add(name + "_sigma")

        # initialize the data frame
        parameter_names = sorted(list(parameter_names))
        if group:
            if "temperature" in parameter_names:
                raise ValueError("'temperature' can not be a fit parameter name if group=True")
            parameter_names.append("temperature")
        if {'chisqr', 'redchi', 'aic', 'bic'}.intersection(set(parameter_names)):
            raise ValueError("'chisqr', 'redchi', 'aic', and 'bic' are reserved and cannot be a fit parameter name.")
        parameter_names += ['chisqr', 'redchi', 'aic', 'bic']
        indices = list(zip(self.powers, self.fields, self.temperature_groups))
        if np.unique(indices, axis=0).shape[0] != len(self.powers):
            log.warning("The data does not have a unique value per table entry")
        multi_index = pd.MultiIndex.from_tuples(indices, names=["power", "field", "temperature"])
        df = pd.DataFrame(np.nan, index=multi_index, columns=parameter_names)
        # fill the data frame
        for index, loop in enumerate(self.loops):
            if parameters[index] is not None:
                for key, parameter in parameters[index].items():
                    df.loc[indices[index]][key] = float(parameter.value)
                    if results[index].errorbars:
                        df.loc[indices[index]][key + "_sigma"] = float(parameter.stderr)
            if group:
                df.loc[indices[index]]["temperature"] = self.temperatures[index]
            df.loc[indices[index]]["chisqr"] = results[index].chisqr
            df.loc[indices[index]]["redchi"] = results[index].redchi
            df.loc[indices[index]]["aic"] = results[index].aic
            df.loc[indices[index]]["bic"] = results[index].bic

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
            loop.sweep = self
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
            loop.free_memory(directory=directory)
        try:
            self._data.free_memory()
        except AttributeError:
            pass

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
                Extra keyword arguments to send to data.
        """
        # create sweep
        sweep = cls()
        # load loop kwargs based on the sweep file
        loop_kwargs_list = data(sweep_file_name, **kwargs)
        loops = []
        # load loops
        for kws in loop_kwargs_list:
            kws.update({"sort": sort})
            loops.append(Loop.load(**kws))
        sweep.add_loops(loops, sort=sort)
        return sweep

    def lmfit(self, parameter, model, guess, index=None, label='default', data_label="best", residual_args=(),
              residual_kwargs=None, **kwargs):
        """
        Compute a least squares fit using the supplied residual function and
        guess. The result and other useful information is stored in
        self.lmfit_results[parameter][label].
        Args:
            parameter: string or list of strings
                The loop parameters to fit. They must be a columns in the loop
                parameters table. If more than one parameter is specified a
                joint fit will be performed and the mkidcalculator.models.Joint
                class should be used.
            model: object-like
                model.residual should give the objective function to minimize.
                It must output a 1D real vector. The first two arguments must
                be a lmfit.Parameters object, and the parameter data. Other
                arguments can be passed in through the residual_args and
                residual_kwargs arguments.
            guess: lmfit.Parameters object
                A parameters object containing starting values (and bounds if
                desired) for all of the parameters needed for the residual
                function.
            index: pandas.IndexSlice (optional)
                A pandas index which specifies which data from the loop
                parameters table should be fit. The default is None and all
                data is fit.
            label: string (optional)
                A label describing the fit, used for storing the results in the
                self.lmfit_results dictionary. The default is 'default'.
            data_label: string (optional)
                The loop parameters table label to use for the fit. The default
                is 'best'.
            residual_args: tuple (optional)
                A tuple of arguments to be passed to the residual function.
                Note: these arguments are the non-mandatory ones after the
                first two. The default is an empty tuple.
            residual_kwargs: dictionary (optional)
                A dictionary of arguments to be passed to the residual
                function. The default is None, which corresponds to an empty
                dictionary.
            kwargs: optional keyword arguments
                Additional keyword arguments are sent to the
                lmfit.Minimizer.minimize() method.
        Returns:
            result: lmfit.MinimizerResult
                An object containing the results of the minimization. It is
                also stored in self.lmfit_results[label]['result'].
        """
        # get the data to fit
        table = self.loop_parameters[data_label] if index is None else self.loop_parameters[data_label].loc[index]
        if isinstance(parameter, str):
            parameter = [parameter]
        args_list = []
        kws_list = []
        # collect the arguments for each parameter
        for p in parameter:
            data = table[p].to_numpy()
            if p == 'fr':
                data = data * 1e9  # convert to Hz for model
            args = (data, *residual_args)
            args_list.append(args)
            if 'temperature' in table.columns:
                temperatures = table['temperature'].to_numpy()
            else:
                temperatures = table.index.get_level_values('temperature')
            powers = table.index.get_level_values("power").to_numpy()
            sigmas = table[p + '_sigma'].to_numpy()
            kws = {"temperatures": temperatures, "powers": powers, "sigmas": sigmas}
            if residual_kwargs is not None:
                kws.update(residual_kwargs)
            kws_list.append(kws)
        # reformat the arguments to work with one or many parameters
        if len(parameter) == 1:
            args = args_list[0]
            kws = kws_list[0]
        else:
            args = [tuple([args_list[ind][index] for ind, _ in enumerate(args_list)])
                    for index, _ in enumerate(args_list[0])]
            kws = {key: tuple(kws_list[ind][key] for ind, _ in enumerate(kws_list)) for key in kws_list[0].keys()}
        # make sure the dictionary exists for each parameter
        for p in parameter:
            if p not in self.lmfit_results.keys():
                self.lmfit_results[p] = {}
        # do the fit for the first parameter
        lmfit(self.lmfit_results[parameter[0]], model, guess, label=label, residual_args=args,
              residual_kwargs=kws, model_index=0 if len(parameter) != 1 else None, **kwargs)
        result = self.lmfit_results[parameter[0]][label]['result']
        # copy the result to the other parameters
        for ind, p in enumerate(parameter[1:]):
            save_lmfit(self.lmfit_results[p], model.models[ind + 1], result, label=label,
                       residual_args=args_list[ind + 1], residual_kwargs=kws_list[ind + 1])
        return result

    def emcee(self):
        raise NotImplementedError

    def fit_report(self, parameter, label='best', fit_type='lmfit', return_string=False):
        """
        Print a string summarizing a sweep fit.
        Args:
            parameter: string
                The parameter on which the fit was done.
            label: string
                The label used to store the fit. The default is "best".
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            return_string: boolean
                Return a string with the fit report instead of printing. The
                default is False.

        Returns:
            string: string
                A string containing the fit report. None is output if
                return_string is False.
        """
        _, result_dict = self._get_model(parameter, fit_type, label)
        string = lm.fit_report(result_dict['result'])
        if return_string:
            return string
        else:
            print(string)

    def _set_directory(self, directory):
        self._directory = directory
        for loop in self.loops:
            loop._set_directory(self._directory)

    def _get_model(self, parameter, fit_type, label):
        if fit_type not in ['lmfit', 'emcee', 'emcee_mle']:
            raise ValueError("'fit_type' must be either 'lmfit', 'emcee', or 'emcee_mle'")
        if fit_type == "lmfit" and label in self.lmfit_results[parameter].keys():
            result_dict = self.lmfit_results[parameter][label]
            original_label = self.lmfit_results[parameter][label]["label"] if label == "best" else label
        elif fit_type == "emcee" and label in self.emcee_results[parameter].keys():
            result_dict = self.emcee_results[parameter][label]
            original_label = self.lmfit_results[parameter][label]["label"] if label == "best" else label
        elif fit_type == "emcee_mle" and label in self.emcee_results[parameter].keys():
            result_dict = copy.deepcopy(self.emcee_results[parameter][label])
            for name in result_dict['result'].params.keys():
                result_dict['result'].params[name].set(value=self.emcee_results[parameter][label]["mle"][name])
            original_label = self.lmfit_results[parameter][label]["label"] if label == "best" else label
        else:
            result_dict = None
            original_label = None
        return original_label, result_dict

    def plot_loops(self, power=None, field=None, temperature=None, color_data='temperature', colormap=None,
                   colorbar=True, colorbar_kwargs=None, colorbar_label=True, colorbar_label_kwargs=None,
                   colorbar_tick_kwargs=None, **loop_kwargs):
        """
        Plot a subset of the loops in the sweep by combining multiple
        loop.plot() calls.
        Args:
            power: tuple of two numbers or tuple of two number tuples
                Inclusive range or ranges of powers to plot. A single number
                will cause only that value to be plotted. The default is to
                include all of the powers.
            field: tuple of two numbers or tuple of two number tuples
                Inclusive range or ranges of fields to plot. A single number
                will cause only that value to be plotted. The default is to
                include all of the fields.
            temperature: tuple of two numbers or tuple of two number tuples
                Inclusive range or ranges of temperatures to plot. A single
                number will cause only that value to be plotted. The default is
                to include all of the temperatures.
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
            colorbar_tick_kwargs: dictionary
                Keyword arguments for the colorbar ticks using
                colorbar_axes.tick_params(). The default is None which uses the
                default options. Keywords in this dictionary override the
                default options.
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
        expand_figure = True if 'axes_list' not in loop_kwargs.keys() else False
        power, field, temperature = create_ranges(power, field, temperature)
        if color_data == 'temperature':
            cmap = matplotlib.cm.get_cmap('coolwarm') if colormap is None else colormap
            cdata = np.array(self.temperatures[::-1]) * 1000
        elif color_data == 'field':
            cmap = matplotlib.cm.get_cmap('viridis') if colormap is None else colormap
            cdata = self.fields[::-1]
        elif color_data == 'power':
            cmap = matplotlib.cm.get_cmap('plasma') if colormap is None else colormap
            cdata = self.powers[::-1]
        else:
            raise ValueError("'{}' is not a valid value of color_data.".format(color_data))
        norm = matplotlib.colors.Normalize(vmin=min(cdata), vmax=max(cdata))
        n_plots = 3 if 'plot_types' not in loop_kwargs.keys() else len(loop_kwargs['plot_types'])
        axes_list = None
        # format title
        title = loop_kwargs.get("title", True)
        if title is True:
            # power
            if len(power) == 1 and np.isinf(power[0]).all():
                title = "All Powers, "
            elif all(x == power[0] for x in power) and power[0][0] == power[0][1]:
                title = "{:.0f} dBm, ".format(power[0][0])
            else:
                title = "({:.0f}, {:.0f}) dBm, ".format(np.min(power[0]), np.max(power[-1]))
            # field
            if len(field) == 1 and np.isinf(field[0]).all():
                title += "All Fields, "
            elif all(x == field[0] for x in field) and field[0][0] == field[0][1]:
                title += "{:.0f} V, ".format(field[0][0])
            else:
                title += "({:.0f}, {:.0f}) V, ".format(np.min(field[0]), np.max(field[-1]))
            # temperature
            if len(temperature) == 1 and np.isinf(temperature[0]).all():
                title += "All Temperatures"
            elif all(x == temperature[0] for x in temperature) and temperature[0][0] == temperature[0][1]:
                title += "{:.0f} mK".format(temperature[0][0] * 1000)
            else:
                title += "({:.0f}, {:.0f}) mK".format(np.min(temperature[0]) * 1000, np.max(temperature[-1]) * 1000)
        # store key word options
        user_plot_kwargs = loop_kwargs.get('plot_kwargs', [])
        if isinstance(user_plot_kwargs, dict):
            user_plot_kwargs = [user_plot_kwargs] * n_plots
        user_data_kwargs = []
        for kw in user_plot_kwargs:
            user_data_kwargs.append(kw.get("data_kwargs", {}))
        user_fit_kwargs = []
        for kw in user_plot_kwargs:
            user_fit_kwargs.append(kw.get("fit_kwargs", {}))
        # make a plot for each loop
        plot_index = 0
        for index, loop in enumerate(self.loops[::-1]):
            condition = (any(power[i][0] <= loop.power <= power[i][1] for i in range(len(power))) and
                         any(field[i][0] <= loop.field <= field[i][1] for i in range(len(field))) and
                         any(temperature[i][0] <= loop.temperature <= temperature[i][1]
                             for i in range(len(temperature))))
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
                    kws.pop('data_kwargs', None)
                    kws.pop('fit_kwargs', None)
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
            kwargs = {'aspect': 30, "pad":-.05, "anchor": (0.5, 1)}
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
                if colorbar_tick_kwargs is not None:
                    cbar.ax.tick_params(**colorbar_tick_kwargs)
            # resize the figure if axes not given directly
            if expand_figure:
                extent = cbar.ax.get_window_extent()
                cbar_width = extent.transformed(axes_list[0].figure.dpi_scale_trans.inverted()).width
                axes_list[0].figure.set_figwidth(axes_list[0].figure.get_figwidth() + cbar_width)
        return axes_list

    def plot_parameters(self, parameters, x="power", data_label="best", n_rows=1, power=None, field=None,
                        temperature=None, plot_fit=False, fit_label="best", axes=None):
        power, field, temperature = create_ranges(power, field, temperature)
        if axes is None:
            from matplotlib import pyplot as plt
            n_columns = int(np.ceil(len(parameters) / n_rows))
            figure, axes = plt.subplots(nrows=n_rows, ncols=n_columns, squeeze=False,
                                        figsize=(4.3 * n_columns, 4.0 * n_rows))
            axes = axes.ravel()
        else:
            if not isinstance(axes, np.ndarray):
                axes = np.atleast_1d(axes)
            figure = axes[0].figure

        levels = ["power", "field", "temperature"]
        if x not in levels:
            raise ValueError("x must be in {}".format(levels))
        levels.remove(x)

        powers = np.unique(self.powers)
        fields = np.unique(self.fields)
        temperatures = np.unique(self.temperature_groups)
        powers = powers[np.logical_and.reduce([(powers >= power[ii][0]) & (powers <= power[ii][1])
                                               for ii in range(len(power))])]
        fields = fields[np.logical_and.reduce([(fields >= field[ii][0]) & (fields <= field[ii][1])
                                               for ii in range(len(field))])]
        temperatures = temperatures[np.logical_and.reduce([(temperatures >= temperature[ii][0]) &
                                                           (temperatures <= temperature[ii][1])
                                                           for ii in range(len(temperature))])]
        values_dict = {"power": powers, "field": fields, "temperature": temperatures}
        table = self.loop_parameters[data_label]

        for index, parameter in enumerate(parameters):
            for ind1, value1 in enumerate(values_dict[levels[0]]):
                for ind2, value2 in enumerate(values_dict[levels[1]]):
                    data = table[parameter].xs((value1, value2), level=levels)
                    if len(data.values):
                        x_vals = data.index
                        if x == "temperature":
                            try:
                                x_vals = table["temperature"].xs((value1, value2), level=levels) * 1000
                            except KeyError:
                                x_vals = data.index * 1000
                        axes[index].plot(x_vals, data.values, 'o')
                        sigma = parameter.split("_")[-1]
                        if sigma == "sigma":
                            axes[index].set_ylabel("_".join(parameter.split("_")[:-1]) + " sigma")
                        else:
                            axes[index].set_ylabel(parameter)
                        x_label = {"power": "power [dBm]", "field": "field [V]", "temperature": "temperature [mK]"}
                        axes[index].set_xlabel(x_label[x])

                        if plot_fit and parameter in self.lmfit_results.keys():
                            if fit_label in self.lmfit_results[parameter].keys():
                                result_dict = self.lmfit_results[parameter][fit_label]
                                result = result_dict['result']
                                model = result_dict['model']
                                parameters = inspect.signature(model.model).parameters
                                residual_kwargs = result_dict['kwargs']
                                kwargs = {}
                                for key in parameters.keys():
                                    if key in residual_kwargs.keys():
                                        kwargs.update({key: residual_kwargs[key]})
                                if 'parallel' in kwargs.keys():
                                    kwargs['parallel'] = bool(kwargs['parallel'])
                                args = result_dict['args'][1:]
                                m = model.model(result.params, *args, **kwargs)
                                x_m = kwargs[x + "s"] if x != "temperature" else 1000 * kwargs[x + "s"]
                                if parameter == "fr":
                                    m *= 1e-9  # Convert to GHz
                                axes[index].plot(x_m, m)
        figure.tight_layout()
