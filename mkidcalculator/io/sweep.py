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

    def create_parameters(self, label="best", fit_type="lmfit", group=True, n_groups=None):
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
                centroids, label = kmeans2(temperatures, k=k, minit='matrix', missing='raise')
            except ClusterError:
                message = "The temperature data is too disordered to cluster into {} groups".format(n_groups)
                raise ClusterError(message)
            self.temperature_groups = np.empty(temperatures.shape)
            for index, centroid in enumerate(centroids):
                self.temperature_groups[label == index] = centroid
            self.temperature_groups = list(self.temperature_groups)
        else:
            self.temperature_groups = self.temperatures
        # determine the parameter names
        parameter_names = set()
        parameters = []
        for loop in self.loops:
            if fit_type == "lmfit" and label in loop.lmfit_results.keys():  # only include lmfit
                p = loop.lmfit_results[label]['result'].params.valuesdict()
            elif fit_type == "emcee" and label in loop.emcee_results.keys():  # only include emcee
                p = loop.emcee_results[label]['median']
            elif fit_type == "emcee_mle" and label in loop.emcee_results.keys():  # only include emcee mle values
                p = loop.emcee_results[label]['mle']
            else:
                p = None
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
        indices = zip(self.powers, self.fields, self.temperature_groups)
        multi_index = pd.MultiIndex.from_tuples(indices, names=["power", "field", "temperature group"])
        df = pd.DataFrame(np.nan, index=multi_index, columns=parameter_names)
        # fill the data frame
        for index, loop in enumerate(self.loops):
            if parameters[index] is not None:
                for key, value in parameters[index].items():
                    df.loc[indices[index]][key] = value
            if group:
                df.loc[indices[index]]["temperature"] = self.temperatures[index]
        self.loop_parameters[label] = df

    def save(self):
        raise NotImplementedError

    def add_loops(self, loops, sort=True):
        """
        Add loop data sets to the sweep.
        Args:
            loops: iterable of Loop() classes
                The loop data sets that are to be added to the Sweep.
            sort: boolean (optional)
                Sort the loop data list by its power, field, and temperature.
                The default is True. If False, the order of the loop data sets
                is preserved.
        """
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

    @classmethod
    def load(cls, sweep_file_name, sweep_data=analogreadout_sweep, sort=True, **kwargs):
        """
        Sweep class factory method that returns a Sweep() with the loop, noise
        and pulse data loaded.
        Args:
            sweep_file_name: string
                The file name for the sweep data. Typically it is a
                configuration file with all the information needed to load in
                the data.
            sweep_data: object (optional)
                Class or function whose return value is a list of dictionaries
                with each being the desired keyword arguments to Loop.load().
            sort: boolean (optional)
                Sort the loop data by its power, field, and temperature. Also
                sort noise data and pulse data lists by their bias frequency.
                The default is True. If False, the input order is preserved.
            kwargs: optional keyword arguments
                Extra keyword arguments to send to Loop.load() not
                specified by sweep_data.
        """
        # create sweep
        sweep = cls()
        # load loop kwargs based on the sweep file
        loop_kwargs_list = sweep_data(sweep_file_name)
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
