import logging
import lmfit as lm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from mkidcalculator.io.noise import Noise
from mkidcalculator.io.pulse import Pulse
from mkidcalculator.io.data import AnalogReadoutLoop, AnalogReadoutNoise, AnalogReadoutPulse

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Loop:
    """A class for manipulating resonance loop scattering parameter data."""
    def __init__(self):
        # loop data
        self._data = AnalogReadoutLoop()  # dummy class replaced by load()
        # noise and pulse classes
        self.noise = []
        self.f_bias_noise = []  # for bias frequency of each noise data set
        self.pulses = []
        self.f_bias_pulses = []  # for bias frequency of each pulse data set
        self.energies_pulses = []  # for known line energies for each pulse data set
        # internal variables
        self._power_calibration = 0
        # analysis results
        self.lmfit_results = {}
        self.emcee_results = {}
        log.info("Loop object created. ID: {}".format(id(self)))

    @property
    def z(self):
        """The complex scattering parameter for the resonance loop."""
        return self._data['z']

    @property
    def f(self):
        """The frequencies corresponding to the complex scattering parameter."""
        return self._data['f']

    @property
    def imbalance_calibration(self):
        """A MxN complex array containing beating IQ mixer data on the rows."""
        return self._data['imbalance']

    @property
    def offset_calibration(self):
        """The mixer offsets corresponding to the complex scattering parameter."""
        return self._data['offset']

    @property
    def power_calibration(self):
        """
        The calibrated power in dBm at the resonator for zero attenuation. This
        value is often unknown so it is initialized to zero. If it is known, it
        can be set directly (e.g. self.power_calibration = value). This value
        is used primarily in the self.power attribute.
        """
        return self._power_calibration

    @power_calibration.setter
    def power_calibration(self, calibration):
        self._power_calibration = calibration

    @property
    def metadata(self):
        """A dictionary containing metadata about the loop."""
        return self._data['metadata']

    @property
    def power(self):
        """
        The power in dBm at the resonator. This value is only relative if
        self.power_calibration has not been set.
        """
        return self.power_calibration - self._data['attenuation']

    @property
    def field(self):
        """The field value at the resonator."""
        return self._data['field']

    @property
    def temperature(self):
        """The temperature at the resonator."""
        return self._data['temperature']

    def add_pulses(self, pulses, sort=True):
        """
        Add pulse data sets to the loop.
        Args:
            pulses: iterable of Pulse() classes
                The pulse data sets that are to be added to the Loop.
            sort: boolean (optional)
                Sort the pulse data list by its bias frequency. The default is
                True. If False, the order of the pulse data sets is preserved.
        """
        # append pulse data
        for p in pulses:
            self.pulses.append(p)
            self.f_bias_pulses.append(p.f_bias)
        # sort
        if sort and self.pulses:
            self.f_bias_pulses, self.pulses = (list(t) for t in zip(*sorted(zip(self.f_bias_pulses, self.pulses))))

    def add_noise(self, noise, sort=True):
        """
        Add noise data sets to the loop.
        Args:
            noise: iterable of Noise() classes
                The noise data sets that are to be added to the Loop.
            sort: boolean (optional)
                Sort the noise data list by its bias frequency. The default is
                True. If False, the order of the noise data sets is preserved.
        """
        # append noise data
        for n in noise:
            self.noise.append(n)
            self.f_bias_noise.append(n.f_bias)
        # sort
        if sort and self.noise:
            self.f_bias_noise, self.noise = (list(t) for t in zip(*sorted(zip(self.f_bias_noise, self.noise))))

    @classmethod
    def load(cls, loop_file_name, noise_file_names=(), pulse_file_names=(), loop_data=AnalogReadoutLoop,
             noise_data=AnalogReadoutNoise, pulse_data=AnalogReadoutPulse, sort=True, channel=None, noise_kwargs=None,
             pulse_kwargs=None, **kwargs):
        """
        Loop class factory method that returns a Loop() with the loop, noise
        and pulse data loaded.
        Args:
            loop_file_name: string
                The file name for the loop data.
            noise_file_names: tuple (optional)
                Tuple of file name strings for the noise data. The default is
                to not load any noise data.
            pulse_file_names: tuple (optional)
                Tuple of file name strings for the pulse data. The default is
                to not load any pulse data.
            loop_data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Loop class. The
                default is the AnalogReadoutLoop class, which interfaces
                with the data products from the analogreadout module.
            noise_data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Noise class. The
                default is the AnalogReadoutNoise class, which interfaces
                with the data products from the analogreadout module.
            pulse_data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Pulse class. The
                default is the AnalogReadoutPulse class, which interfaces
                with the data products from the analogreadout module.
            sort: boolean (optional)
                Sort the noise data and pulse data lists by their bias
                frequency. The default is True. If False, the order of the
                noise and pulse file names is preserved.
            channel: integer (optional)
                Optional channel index which gets added to all of the kwarg
                dictionaries under the key 'channel'. When the default, None,
                is passed, nothing is added to the dictionaries.
            noise_kwargs: tuple (optional)
                Tuple of dictionaries for extra keyword arguments to send to
                noise_data. The order and length correspond to
                noise_file_names. The default is None, which is equivalent to
                a tuple of empty dictionaries.
            pulse_kwargs: tuple (optional)
                Tuple of dictionaries for extra keyword arguments to send to
                pulse_data. The order and length correspond to
                pulse_file_names. The default is None, which is equivalent to
                a tuple of empty dictionaries.
            kwargs: optional keyword arguments
                extra keyword arguments to send to loop_data.
        Returns:
            loop: object
                A Loop() object containing the loaded data.
        """
        # create loop
        loop = cls()
        # update dictionaries
        if noise_kwargs is None:
            noise_kwargs = [{} for _ in range(len(noise_file_names))]
        if pulse_kwargs is None:
            pulse_kwargs = [{} for _ in range(len(pulse_file_names))]
        if channel is not None:
            kwargs.update({"channel": channel})
            for kws in noise_kwargs:
                kws.update({"channel": channel})
            for kws in pulse_kwargs:
                kws.update({"channel": channel})
        # load loop
        loop._data = loop_data(loop_file_name, **kwargs)
        # load noise
        noise = []
        for index, noise_file_name in enumerate(noise_file_names):
            noise.append(Noise.load(noise_file_name, data=noise_data, **noise_kwargs[index]))
        loop.add_noise(noise, sort=sort)
        # load pulses
        pulses = []
        for index, pulse_file_name in enumerate(pulse_file_names):
            pulses.append(Pulse.load(pulse_file_name, data=pulse_data, **pulse_kwargs[index]))
        loop.add_pulses(pulses, sort=sort)
        return loop

    def to_pickle(self):
        raise NotImplementedError

    def from_pickle(self):
        raise NotImplementedError

    def lmfit(self, model, guess, label='default', residual_args=(), residual_kwargs=None, **kwargs):
        """
        Compute a least squares fit using the supplied residual function and
        guess. The result and other useful information is stored in
        self.lmfit_results[label].
        Args:
            model: module or object-like
                model.residual should give the objective function to minimize.
                It must output a 1D real vector. The first three arguments must
                be a lmfit.Parameters object, the complex scattering parameter,
                and the corresponding frequencies. Other arguments can be
                passed in through the residual_args and residual_kwargs
                arguments.
            guess: lmfit.Parameters object
                A parameters object containing starting values (and bounds if
                desired) for all of the parameters needed for the residual
                function.
            label: string (optional)
                A label describing the fit, used for storing the results in the
                self.lmfit_results dictionary. The default is 'default'.
            residual_args: tuple (optional)
                A tuple of arguments to be passed to the residual function.
                Note: these arguments are the non-mandatory ones after the
                first three. The default is an empty tuple.
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
        Raises:
            ValueError:
                Improper arguments or keyword arguments.
        """
        if label == 'best':
            raise ValueError("'best' is a reserved label and cannot be used")
        # set up and do minimization
        residual_args = (self.z, self.f, *residual_args)
        minimizer = lm.Minimizer(model.residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
        result = minimizer.minimize(**kwargs)
        # save the results
        self.lmfit_results[label] = {'result': result, 'model': model}
        # if the result is better than has been previously computed, add it to the 'best' key
        if 'best' not in self.lmfit_results.keys():
            self.lmfit_results['best'] = self.lmfit_results[label]
            self.lmfit_results['best']['label'] = label
        elif result.aic < self.lmfit_results['best']['result'].aic:
            self.lmfit_results['best'] = self.lmfit_results[label]
            self.lmfit_results['best']['label'] = label
        return result

    def emcee(self, model, label='default', residual_args=(), residual_kwargs=None, **kwargs):
        """
        Compute a MCMC using the supplied log likelihood function. The result
        and other useful information is stored in self.emcee_results[label].
        Args:
            model: module or object-like
                model.residual should give the objective function to minimize.
                It must output a 1D real vector. The first three arguments must
                be a lmfit.Parameters object, the complex scattering parameter,
                and the corresponding frequencies. Other arguments can be
                passed in through the residual_args and residual_kwargs
                arguments.
            label: string (optional)
                A label describing the fit, used for storing the results in the
                self.emcee_results dictionary. A corresponding fit must already
                exist in the self.lmfit_results dictionary. The default is
                'default'.
            residual_args: tuple (optional)
                A tuple of arguments to be passed to the residual function.
                Note: these arguments are the non-mandatory ones after the
                first three. The default is an empty tuple.
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
                also stored in self.emcee_results[label]['result'].
        Raises:
            ValueError:
                Improper arguments or keyword arguments.
        """
        if label == 'best':
            raise ValueError("'best' is a reserved label and cannot be used")
        elif label not in self.lmfit_results.keys():
            raise ValueError("The MCMC cannot be run unless an lmfit using this label has run first.")
        # set up and do MCMC
        residual_args = (self.z, self.f, *residual_args)
        guess = self.lmfit_results[label]['result'].params
        minimizer = lm.Minimizer(model.residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
        result = minimizer.minimize(method='emcee', **kwargs)
        # get the MLE, median, and 1 sigma uncertainties around the median for each parameter in the flatchain
        one_sigma = 1 - 2 * stats.norm.cdf(-1)
        p = (100 - one_sigma) / 2
        median = {key: np.percentile(result.flatchain[key], 50) for key in result.flatchain.keys()}
        sigma = {key: (np.percentile(result.flatchain[key], p), np.percentile(result.flatchain[key], 100 - p))
                 for key in result.flatchain.keys()}
        mle = dict(result.flatchain.iloc[np.argmax(result.lnprob)])
        # save the results
        self.emcee_results[label] = {'result': result, 'model': model, 'median': median, 'sigma': sigma, 'mle': mle}
        # if the result is better than has been previously computed, add it to the 'best' key
        if 'best' not in self.emcee_results.keys():
            self.emcee_results['best'] = self.lmfit_results[label]
            self.emcee_results['best']['label'] = label
        elif result.aic < self.emcee_results['best']['result'].aic:
            self.emcee_results['best']['result'] = result
            self.emcee_results['best']['objective'] = residual
            self.emcee_results['best']['label'] = label
        return result

    def plot_iq(self, data_kwargs=None, plot_fit=False, fit_label="best", fit_type="lmfit", fit_kwargs=None,
                x_label=None, y_label=None, label_kwargs=None, legend=True, legend_kwargs=None, title=True,
                title_kwargs=None, axes=None):
        # parse inputs
        if axes is None:
            _, axes = plt.subplots()
        if x_label is None:
            x_label = "I [V]"
        if y_label is None:
            y_label = "Q [V]"
        if fit_type not in ['lmfit', 'emcee', 'emcee_mle']:
            raise ValueError("'fit_type' must be either 'lmfit', 'emcee', or 'emcee_mle'")
        # setup axes
        axes.axis('equal')
        kwargs = {}
        if label_kwargs is not None:
            kwargs.update(label_kwargs)
        axes.set_xlabel(x_label, **kwargs)
        axes.set_ylabel(y_label, **kwargs)
        # plot data
        kwargs = {"marker": 'o', "markersize": 3, "linestyle": 'None', "label": "data"}
        if data_kwargs is not None:
            kwargs.update(data_kwargs)
        axes.plot(self.z.real, self.z.imag, **kwargs)
        # plot fit
        if plot_fit:
            # get the model
            if fit_type == "lmfit":
                model = self.lmfit_results[fit_label]["model"]
                params = self.lmfit_results[fit_label]["result"].params
                fit_name = self.lmfit_results[fit_label]["label"] if fit_label == "best" else fit_label
            elif fit_type == "emcee":
                model = self.emcee_results[fit_label]["model"]
                params = self.emcee_results[fit_label]["result"].params
                fit_name = self.lmfit_results[fit_label]["label"] if fit_label == "best" else fit_label
            else:
                model = self.emcee_results[fit_label]["model"]
                params = self.emcee_results[fit_label]["result"].params.copy()  # copy(): don't change existing params
                for name in params.keys():
                    params['name'].set(value=self.emcee_results[fit_label]["mle"][name])
                fit_name = self.lmfit_results[fit_label]["label"] if fit_label == "best" else fit_label
            # calculate the model values
            f = np.linspace(np.min(self.f), np.max(self.f), np.size(self.f) * 10)
            m = model.model(params, f)
            # add the plot
            kwargs = {"linestyle": '--', "label": "fit"}
            if fit_kwargs is not None:
                kwargs.update(fit_kwargs)
            axes.plot(m.real, m.imag, **kwargs)
            label = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK, '{}' fit"
            title = label.format(self.power, self.field, self.temperature * 1000, fit_name) if title is True else title
        else:
            label = "power: {:.0f} dBm, field: {:.2f} V, temperature: {:.2f} mK"
            title = label.format(self.power, self.field, self.temperature * 1000) if title is True else title
        if legend:
            kwargs = {}
            if legend_kwargs is not None:
                kwargs.update(legend_kwargs)
            axes.legend(**kwargs)
        if title:
            kwargs = {"fontsize": 11}
            if title_kwargs is not None:
                kwargs.update(title_kwargs)
            axes.set_title(title, **kwargs)
        axes.figure.tight_layout()
