import pickle
import logging

from mkidcalculator.io.data import AnalogReadoutPulse

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Pulse:
    """A class for manipulating the pulse data."""
    def __init__(self):
        self._data = AnalogReadoutPulse()  # dummy class replaced by load()
        log.info("Pulse object created. ID: {}".format(id(self)))

    @property
    def f_bias(self):
        """The bias frequency for the data set."""
        return self._data["f_bias"]

    @property
    def i_trace(self):
        """The mixer I output traces."""
        return self._data["i_trace"]

    @property
    def q_trace(self):
        """The mixer Q output traces."""
        return self._data["q_trace"]

    @property
    def offset(self):
        """The mixer IQ offset at the bias frequency."""
        return self._data["offset"]

    @property
    def metadata(self):
        """A dictionary containing metadata about the pulse."""
        return self._data["metadata"]

    @property
    def energies(self):
        """The known photon energies in this data set."""
        return self._data["energies"]

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Pulse class from the pickle file 'file_name'."""
        with open(file_name, "rb") as f:
            pulse = pickle.load(f)
        assert isinstance(pulse, cls), "'{}' does not contain a Pulse class.".format(file_name)
        return pulse

    @classmethod
    def load(cls, file_name, data=AnalogReadoutPulse, **kwargs):
        """
        Pulse class factory method that returns a Pulse() with the data loaded.
        Args:
            file_name: string
                The file name for the pulse data.
            data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Pulse class. The
                default is the AnalogReadoutPulse class, which interfaces
                with the data products from the analogreadout module.
            kwargs: optional keyword arguments
                extra keyword arguments are sent to 'data'. This is useful in
                the case of the AnalogReadout* data classes for picking the
                channel and index.
        Returns:
            pulse: object
                A Pulse() object containing the loaded data.
        """
        pulse = cls()
        pulse._data = data(file_name, **kwargs)
        return pulse

    def compute_photon_energies(self):
        raise NotImplementedError
