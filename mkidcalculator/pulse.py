import logging

from mkidcalculator.data import AnalogReadoutPulse

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Pulse:
    def __init__(self):
        self._data = AnalogReadoutPulse()  # dummy class replaced by load()
        log.info("Pulse object created. ID: {}".format(id(self)))

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
    def offset(self):
        return self._data["offset"]

    @property
    def metadata(self):
        return self._data["metadata"]

    @classmethod
    def load(cls, file_name, data=AnalogReadoutPulse, **kwargs):
        pulse = cls()
        pulse._data = data(file_name, **kwargs)
        return pulse

    def compute_energies(self):
        raise NotImplementedError
