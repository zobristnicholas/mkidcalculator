import pickle
import logging

from mkidcalculator.io.data import AnalogReadoutNoise
from mkidcalculator.io.utils import compute_phase_and_amplitude

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Noise:
    """A class for manipulating the noise data."""
    def __init__(self):
        self._data = AnalogReadoutNoise()  # dummy class replaced by load()
        # loop reference for computing phase and amplitude
        self._loop = None
        # phase and amplitude data
        self._phase_trace = None
        self._amplitude_trace = None
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
        """A dictionary containing metadata about the noise."""
        return self._data["metadata"]

    @property
    def attenuation(self):
        """The DAC attenuation used for the data set."""
        return self._data['attenuation']

    @property
    def sample_rate(self):
        """The sample rate of the IQ data."""
        return self._data['sample_rate']

    @property
    def loop(self):
        """
        A settable property that contains the Loop object required for doing
        noise calculations like computing the phase and amplitude traces. If
        the loop has not been set, it will raise an AttributeError. When the
        loop is set, all information created from the previous loop is deleted.
        """
        if self._loop is None:
            raise AttributeError("The loop object for this noise has not been set yet.")
        return self._loop

    @loop.setter
    def loop(self, loop):
        self._loop = loop
        self.clear_loop_data()

    @property
    def phase_trace(self):
        """
        A settable property that contains the phase trace information. Since it
        is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() is run.
        """
        if self._phase_trace is None:
            raise AttributeError("The phase information has not been computed yet.")
        return self._phase_trace

    @phase_trace.setter
    def phase_trace(self, phase_trace):
        self._phase_trace = phase_trace

    @property
    def amplitude_trace(self):
        """
        A settable property that contains the amplitude trace information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() is run.
        """
        if self._amplitude_trace is None:
            raise AttributeError("The amplitude information has not been computed yet.")
        return self._amplitude_trace

    @amplitude_trace.setter
    def amplitude_trace(self, amplitude_trace):
        self._amplitude_trace = amplitude_trace

    def clear_loop_data(self):
        """Remove all data calculated from the noise.loop attribute."""
        self.amplitude_trace = None
        self.phase_trace = None

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, file_name):
        """Returns a Noise class from the pickle file 'file_name'."""
        with open(file_name, "rb") as f:
            noise = pickle.load(f)
        assert isinstance(noise, cls), "'{}' does not contain a Noise class.".format(file_name)
        return noise

    @classmethod
    def load(cls, noise_file_name, data=AnalogReadoutNoise, **kwargs):
        """
        Noise class factory method that returns a Noise() with the data loaded.
        Args:
            noise_file_name: string
                The file name for the noise data.
            data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Noise class. The
                default is the AnalogReadoutNoise class, which interfaces
                with the data products from the analogreadout module.
            kwargs: optional keyword arguments
                extra keyword arguments are sent to 'data'. This is useful in
                the case of the AnalogReadout* data classes for picking the
                channel and index.
        Returns:
            noise: object
                A Noise() object containing the loaded data.
        """
        noise = cls()
        noise._data = data(noise_file_name, **kwargs)
        return noise

    def compute_phase_and_amplitude(self, label="best", fit_type="lmfit", fr="fr", center="center", unwrap=True):
        """
        Compute the phase and amplitude traces stored in noise.phase_trace and
        noise.amplitude_trace.
        Args:
            label: string
                Corresponds to the label in the loop.lmfit_results or
                loop.emcee_results dictionaries where the fit parameters are.
                The resulting DataFrame is stored in
                self.loop_parameters[label]. The default is "best", which gets
                the parameters from the best fits.
            fit_type: string
                The type of fit to use. Allowed options are "lmfit", "emcee",
                and "emcee_mle" where MLE estimates are used instead of the
                medians. The default is "lmfit".
            fr: string
                The parameter name that corresponds to the resonance frequency.
                The default is "fr". This parameter determines the zero point
                for the traces.
            center: string
                The parameter name that corresponds to the calibrated loop
                center. The default is "center".
            unwrap: boolean
                Determines whether or not to unwrap the phase data. The default
                is True.
        """
        compute_phase_and_amplitude(self, label=label, fit_type=fit_type, fr=fr, center=center, unwrap=unwrap)

    def make_psd(self, psd_type='iq'):
        raise NotImplementedError
