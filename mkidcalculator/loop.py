import logging
import scipy.stats as stats

from mkidcalculator.noise import Noise
from mkidcalculator.pulse import Pulse
from mkidcalculator.data import AnalogReadoutLoop, AnalogReadoutNoise, AnalogReadoutPulse

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
        # analysis results
        self.lmfit_results = {'best': None}
        self.emcee_results = {'best': None}
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
    def metadata(self):
        """A dictionary containing metadata about the loop."""
        return self._data['metadata']

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
            noise_file_names: tuple
                Tuple of file name strings for the noise data. The default is
                to not load any noise data.
            pulse_file_names: tuple
                Tuple of file name strings for the pulse data. The default is
                to not load any pulse data.
            loop_data: object
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Loop class. The
                default is the AnalogReadoutLoop class, which interfaces
                with the data products from the analogreadout module.
            noise_data: object
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Noise class. The
                default is the AnalogReadoutNoise class, which interfaces
                with the data products from the analogreadout module.
            pulse_data: object
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Pulse class. The
                default is the AnalogReadoutPulse class, which interfaces
                with the data products from the analogreadout module.
            sort: boolean
                Sort the noise data and pulse data lists by their bias
                frequency. The default is True. If False, the order of the
                noise and pulse file names is preserved.
            channel: integer
                Optional channel index which gets added to all of the kwarg
                dictionaries under the key 'index'. When the default, None, is
                passed, nothing is added to the dictionaries.
            noise_kwargs: tuple
                Tuple of dictionaries for extra keyword arguments to send to
                noise_data. The order and length correspond to
                noise_file_names. The default is None, which is equivalent to
                a tuple of empty dictionaries.
            pulse_kwargs: tuple
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
        # load noise and pulses
        for index, noise_file_name in enumerate(noise_file_names):
            loop.noise.append(Noise.load(noise_file_name, data=noise_data, **noise_kwargs[index]))
        for index, pulse_file_name in enumerate(pulse_file_names):
            loop.pulses.append(Pulse.load(pulse_file_name, data=pulse_data, **pulse_kwargs[index]))
        # pull out the bias frequencies
        for n in loop.noise:
            loop.f_bias_noise.append(n.f_bias)
        for p in loop.pulses:
            loop.f_bias_pulses.append(p.f_bias)
        # sort the noise and pulses
        if sort and loop.noise:
            loop.f_bias_noise, loop.noise = (list(t) for t in zip(*sorted(zip(loop.f_bias_noise, loop.noise))))
        if sort and loop.pulses:
            loop.f_bias_pulses, loop.pulses = (list(t) for t in zip(*sorted(zip(loop.f_bias_pulses, loop.pulses))))
        return loop

    def to_pickle(self):
        raise NotImplementedError

    def from_pickle(self):
        raise NotImplementedError

    def lmfit(self, residual, guess, label='default', residual_args=None, residual_kwargs=None, **kwargs):
        """
        Compute a least squares fit using the supplied residual function and
        guess. The result and other useful information is stored in
        self.lmfit_results[label].
        Args:
            residual: function
                The objective function to minimize. It must output a 1D real
                vector. The first three arguments must be a lmfit.Parameters
                object, the complex scattering parameter, and the corresponding
                frequencies. Other arguments can be passed in through the
                residual_args and residual_kwargs arguments.
            guess: lmfit.Parameters object
                A parameters object containing starting values (and bounds if
                desired) for all of the parameters needed for the residual
                function.
            label: string
                A label describing the fit, used for storing the results in the
                self.lmfit_results dictionary.
            residual_args: tuple
                A tuple of arguments to be passed to the residual function.
            residual_kwargs:
                A dictionary of arguments to be passed to the residual
                function.
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
        minimizer = lm.Minimizer(residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
        result = minimizer.minimize(**kwargs)
        # save the results
        self.lmfit_results[label] = {'result': result, 'objective': residual}
        # if the result is better than has been previously computed, add it to the 'best' key
        if self.lmfit_results['best'] is None:
            self.lmfit_results['best'] = self.lmfit_results[label]
            self.lmfit_results['best']['label'] = label
        elif result.aic < self.lmfit_results['best']['result'].aic:
            self.lmfit_results['best']['result'] = result
            self.lmfit_results['best']['objective'] = residual
            self.lmfit_results['best']['label'] = label
        return result

    def emcee(self, residual, label='default', residual_args=None, residual_kwargs=None, **kwargs):
        """
        Compute a MCMC using the supplied log likelihood function. The result
        and other useful information is stored in self.emcee_results[label].
        Args:
            residual: function
                The objective function to minimize. It must output a 1D real
                vector. The first three arguments must be a lmfit.Parameters
                object, the complex scattering parameter, and the corresponding
                frequencies. Other arguments can be passed in through the
                residual_args and residual_kwargs arguments.
            label: string
                A label describing the fit, used for storing the results in the
                self.emcee_results dictionary. A corresponding fit must already
                exist in the self.lmfit_results dictionary.
            residual_args: tuple
                A tuple of arguments to be passed to the residual function.
            residual_kwargs:
                A dictionary of arguments to be passed to the residual
                function.
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
        minimizer = lm.Minimizer(residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
        result = minimizer.minimize(method='emcee', **kwargs)
        # get the MLE, median, and 1 sigma uncertainties around the median for each parameter in the flatchain
        one_sigma = 1 - 2 * stats.norm.cdf(-1)
        p = (100 - one_sigma) / 2
        median = {key: np.percentile(result.flatchain[key], 50) for key in result.flatchain.keys()}
        sigma = {key: (np.percentile(result.flatchain[key], p), np.percentile(result.flatchain[key], 100 - p))
                 for key in result.flatchain.keys()}
        mle = dict(emcee_result.flatchain.iloc[np.argmax(emcee_result.lnprob)])
        # save the results
        self.emcee_results[label] = {'result': result, 'objective': residual, 'median': median, 'sigma': sigma,
                                     'mle': mle}
        # if the result is better than has been previously computed, add it to the 'best' key
        if self.emcee_results['best'] is None:
            self.emcee_results['best'] = self.lmfit_results[label]
            self.emcee_results['best']['label'] = label
        elif result.aic < self.emcee_results['best']['result'].aic:
            self.emcee_results['best']['result'] = result
            self.emcee_results['best']['objective'] = residual
            self.emcee_results['best']['label'] = label
        return result
