import logging

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
        self.f_bias_noise = []
        self.pulses = []
        self.f_bias_pulses = []
        self.energy_pulses = []
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
        return self._data['freqs']

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
             noise_data=AnalogReadoutNoise, pulse_data=AnalogReadoutPulse, sort=True, **kwargs):
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
            kwargs:
                extra keyword arguments are sent to loop_data, noise_data, and
                pulse_data. This is useful in the case of the AnalogReadout*
                data classes for picking the channel index.
        Returns:
            loop: object
                A Loop() object containing the loaded data.
        """
        loop = cls()
        # load loop
        loop._data = loop_data(loop_file_name, **kwargs)
        # load noise and pulses
        for noise_file_name in noise_file_names:
            loop.noise.append(Noise.load(noise_file_name, data=noise_data, **kwargs))
        for pulse_file_name in pulse_file_names:
            loop.pulses.append(Pulse.load(pulse_file_name, data=pulse_data, **kwargs))
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

    def lmfit(self, residual, guess, label='default', residual_args=None, residual_kwargs=None):
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
        Returns:
            result: lmfit.MinimizerResult
                An object containing the results of the minimization. It is
                also stored in self.lmfit_results[label]['result'].
        """
        residual_args = (self.z, self.f, *residual_args)
        minimizer = lm.Minimizer(residual, guess, fcn_args=residual_args, fcn_kws=residual_kwargs)
        result = minimizer.minimize(method='leastsq')
        self.lmfit_results[label] = {'result': result, 'objective': residual}
        if self.lmfit_results['best'] is None:
            self.lmfit_results['best'] = self.lmfit_results[label]
        elif result.aic < self.lmfit_results['best']['result']:
            self.lmfit_results['best']['result'] = result
            self.lmfit_results['best']['objective'] = residual
        return result

    def emcee(self, log_likelihood, guess, label='default', **kwargs):
        raise NotImplementedError
