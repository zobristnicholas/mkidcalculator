import logging

from mkidcalculator.data import AnalogReadoutNoise

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Noise:
    def __init__(self):
        self._data = AnalogReadoutNoise()  # dummy class replaced by load()
        log.info("Noise object created. ID: {}".format(id(self)))

    @property
    def f_bias(self):
        return self._data["f_bias"]

    @property
    def i_trace(self):
        return self._data["i_trace"]

    @property
    def q_trace(self):
        return self._data["q_trace"]

    @property
    def metadata(self):
        return self._data["metadata"]

    @classmethod
    def load(cls, file_name, data=AnalogReadoutNoise, **kwargs):
        noise = cls()
        noise._data = data(file_name, **kwargs)
        return noise

    def compute_phase_and_amplitude(self):
        raise NotImplementedError

    def make_psd(self, psd_type='iq'):
        raise NotImplementedError
