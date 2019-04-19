import logging

from mkidcalculator.data import AnalogReadoutNoise

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Noise:
    def __init__(self):
        self._data = AnalogReadoutNoise()  # dummy class replaced by load()
        log.info("Noise object created. ID: {}".format(id(self)))

    @parameter
    def f_bias(self):
        return self._data["f_bias"]

    @parameter
    def i_trace(self):
        return self._data["i_trace"]

    @parameter
    def q_trace(self):
        return self._data["q_trace"]

    @parameter
    def metadata(self):
        return self._data["metadata"]

    @classmethod
    def load(cls, file_name, data_class=AnalogReadoutNoise, **kwargs):
        noise = cls()
        noise._data = data_class(file_name, **kwargs)
        return noise

    def compute_phase_and_amplitude(self):
        raise NotImplementedError

    def make_psd(self, psd_type='iq'):
        raise NotImplementedError
