import pickle
import logging
import numpy as np

from mkidcalculator.io.data import AnalogReadoutPulse

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Pulse:
    """A class for manipulating the pulse data."""
    def __init__(self):
        # pulse data
        self._data = AnalogReadoutPulse()  # dummy class replaced by load()
        # loop reference for computing phase and amplitude
        self._loop = None
        # noise reference for computing energies
        self._noise = None
        # phase and amplitude data
        self._phase_trace = None
        self._amplitude_trace = None
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
    def attenuation(self):
        """The DAC attenuation used for the data set."""
        return self._data['attenuation']

    @property
    def energies(self):
        """The known photon energies in this data set."""
        return self._data["energies"]

    @property
    def phase_trace(self):
        """
        A settable property that contains the phase trace information. Since it
        is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        pulse.compute_phase_and_amplitude() is run.
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
        pulse.compute_phase_and_amplitude() is run.
        """
        if self._amplitude_trace is None:
            raise AttributeError("The amplitude information has not been computed yet.")
        return self._amplitude_trace

    @amplitude_trace.setter
    def amplitude_trace(self, amplitude_trace):
        self._amplitude_trace = amplitude_trace

    @property
    def loop(self):
        """
        A settable property that contains the Loop object required for doing
        pulse calculations like computing the phase and amplitude traces. If
        the loop has not been set, it will raise an AttributeError. When the
        loop is set, all information created from the previous loop is deleted.
        """
        if self._loop is None:
            raise AttributeError("The loop object for this pulse has not been set yet.")
        return self._loop

    @loop.setter
    def loop(self, loop):
        self._loop = loop
        self.clear_loop_data()

    @property
    def noise(self):
        """
        A settable property that contains the Noise object required for doing
        pulse calculations like optimal filtering. If the noise has not been
        set, it will raise an AttributeError. When the noise is set, all
        information created from the previous noise is deleted.
        """
        if self._noise is None:
            raise AttributeError("The noise object for this pulse has not been set yet.")
        return self._noise

    @noise.setter
    def noise(self, noise):
        self._noise = noise
        self.clear_noise_data()

    def clear_loop_data(self):
        """Remove all data calculated from the pulse.loop attribute."""
        self.amplitude_trace = None
        self.phase_trace = None

    def clear_noise_data(self):
        pass

    def compute_phase_and_amplitude(self, label="best", fit_type="lmfit", radius="q0 / (2 * qc)",
                                    fr="fr", unwrap=True):
        """
        Compute the phase and amplitude traces stored in pulse.phase_trace and
        pulse.amplitude_trace.
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
            radius: string
                An evaluable expression of parameters that corresponds to the
                loop radius. The default is "q0 / (2 * qc)".
            fr: string
                An evaluable expression of parameters that corresponds to the
                loop radius. The default is "fr".
            unwrap: boolean
                Determines whether or not to unwrap the phase data. The default
                is True.
        """
        # get the model and parameters
        _, result_dict = self.loop._get_model(fit_type, label)
        model = result_dict["model"]
        params = result_dict["result"].params
        # get the resonance frequency and normalized loop radius
        fr = params._asteval.eval(fr)
        r = params._asteval.eval(radius)
        # get complex IQ data for the traces and loop at the resonance frequency
        traces = self.i_trace + 1j * self.q_trace
        z_fr = model.model(params, fr)
        f = np.empty(traces.shape)
        f.fill(self.f_bias)
        # calibrate the IQ data
        traces = model.calibrate(params, traces, f)
        z_fr = model.calibrate(params, z_fr, fr)
        # center and rotate the IQ data
        centered_trace = 1 - r - traces
        centered_z_f0 = 1 - r - z_fr  # should be real if no loop rotation
        # compute the phase and amplitude traces from the centered traces
        phase_trace = np.angle(centered_trace) - np.angle(centered_z_f0)
        self.phase_trace = np.unwrap(phase_trace) if unwrap else phase_trace
        self.amplitude_trace = np.abs(centered_trace) - r

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
    def load(cls, pulse_file_name, data=AnalogReadoutPulse, loop=None, noise=None, **kwargs):
        """
        Pulse class factory method that returns a Pulse() with the data loaded.
        Args:
            pulse_file_name: string
                The file name for the pulse data.
            data: object (optional)
                Class or function whose return value allows dictionary-like
                queries of the attributes required by the Pulse class. The
                default is the AnalogReadoutPulse class, which interfaces
                with the data products from the analogreadout module.
            loop: Loop object (optional)
                The Loop object needed for computing phase and amplitude. It
                can be specified later or changed with pulse.loop = loop. The
                default is None, which signifies that the loop has not been
                set.
            noise: Noise object (optional)
                The Noise object needed for computing the pulse energies. It
                can be specified later or changed with pulse.noise = noise. The
                default is None, which signifies that the noise has not been
                set.
            kwargs: optional keyword arguments
                extra keyword arguments are sent to 'data'. This is useful in
                the case of the AnalogReadout* data classes for picking the
                channel and index.
        Returns:
            pulse: object
                A Pulse() object containing the loaded data.
        """
        pulse = cls()
        pulse._data = data(pulse_file_name, **kwargs)
        if loop is not None:  # don't set loop unless needed.
            pulse.loop = loop
        if noise is not None:
            pulse.noise = noise
        return pulse

    def compute_trace_energies(self):
        raise NotImplementedError

    def plot_traces(self, calibrated=False, label="best", fit_type="lmfit"):
        pass
