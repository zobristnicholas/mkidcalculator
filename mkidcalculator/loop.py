import logging

from mkidcalculator.noise import Noise
from mkidcalculator.pulse import Pulse
from mkidcalculator.data import AnalogReadoutLoop, AnalogReadoutNoise, AnalogReadoutPulse

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Loop:
    def __init__(self):
        # loop data
        self._data = AnalogReadoutLoop()  # dummy class replaced by load()
        # noise and pulse classes
        self.noise = []
        self.f_bias_noise = []
        self.pulses = []
        self.f_bias_pulses = []
        # analysis results
        self.lmfit_results = {'best': None}
        self.emcee_results = {'best': None}
        log.info("Loop object created. ID: {}".format(id(self)))

    @parameter
    def z(self):
        return self._data['z']

    @parameter
    def f(self):
        return self._data['freqs']

    @parameter
    def imbalance_calibration(self):
        return self._data['imbalance']

    @parameter
    def offset_calibration(self):
        return self._data['offset']

    @parameter
    def metadata(self):
        return self._data['metadata']

    @classmethod
    def load(cls, loop_file_name, noise_file_names=(), pulse_file_names=(), loop_data_class=AnalogReadoutLoop,
             noise_data_class=AnalogReadoutNoise, pulse_data_class=AnalogReadoutPulse, sort=True, **kwargs):
        loop = cls()
        # load loop
        loop._data = loop_data_class(loop_file_name, **kwargs)
        # load noise and pulses
        for noise_file_name in noise_file_names:
            loop.noise.append(Noise.load(noise_file_name, data_class=noise_data_class, **kwargs))
        for pulse_file_name in pulse_file_names:
            loop.pulses.append(Pulse.load(pulse_file_name, data_class=pulse_data_class, **kwargs))
        # pull out the bias frequencies
        for n in loop.noise:
            loop.f_bias_noise.append(n.f_bias)
        for p in loop.pulses:
            loop.f_bias_pulses.append(p.f_bias)
        # sort the noise and pulses
        if sort:
            loop.noise, loop.f_bias_noise = (list(t) for t in zip(*sorted(zip(loop.noise, loop.f_bias_noise))))
            loop.pulses, loop.f_bias_pulses = (list(t) for t in zip(*sorted(zip(loop.pulses, loop.f_bias_pulses))))
        return loop

    def to_pickle(self):
        raise NotImplementedError

    def from_pickle(self):
        raise NotImplementedError

    def lmfit(self, residual, guess, label='default', **kwargs):
        raise NotImplementedError

    def emcee(self, log_likelihood, guess, label='default', **kwargs):
        raise NotImplementedError
