import logging

from mkidcalculator.data import AnalogReadoutPulse

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Pulse:
    def __init__(self):
        self._data = AnalogReadoutPulse()  # dummy class replaced by load()
        log.info("Pulse object created. ID: {}".format(id(self)))

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
    def offset(self):
        return self._data["offset"]

    @parameter
    def metadata(self):
        return self._data["metadata"]

    @classmethod
    def load(cls, file_name, data_class=AnalogReadoutPulse, **kwargs):
        pulse = cls()
        pulse._data = data_class(file_name, **kwargs)
        return pulse

    def compute_energies(self):
        raise NotImplementedError
