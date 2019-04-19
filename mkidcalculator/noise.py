import logging

from mkidcalculator.data import AnalogReadoutNoise

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Noise:
    """A class for manipulating the noise data."""
    def __init__(self):
        self._data = AnalogReadoutNoise()  # dummy class replaced by load()
        log.info("Noise object created. ID: {}".format(id(self)))

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
    def metadata(self):
        """A dictionary containing metadata about the pulse."""
        return self._data["metadata"]

    @classmethod
    def load(cls, file_name, data=AnalogReadoutNoise, **kwargs):
        """
        Noise class factory method that returns a Noise() with the data loaded.
        Args:
            file_name: string
                The file name for the noise data.
            data: object
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Noise class. The
                default is the AnalogReadoutNoise class, which interfaces
                with the data products from the analogreadout module.
            kwargs: optional keyword arguments
                extra keyword arguments are sent to 'data'. This is useful in
                the case of the AnalogReadout* data classes for picking the
                channel index.
        Returns:
            noise: object
                A Noise() object containing the loaded data.
        """
        noise = cls()
        noise._data = data(file_name, **kwargs)
        return noise

    def compute_phase_and_amplitude(self):
        raise NotImplementedError

    def make_psd(self, psd_type='iq'):
        raise NotImplementedError
