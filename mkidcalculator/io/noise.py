import os
import pickle
import logging
import numpy as np
from scipy.signal import welch, csd

from mkidcalculator.io.data import AnalogReadoutNoise
from mkidcalculator.io.utils import compute_phase_and_amplitude, offload_data, _loaded_npz_files

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class Noise:
    """A class for manipulating the noise data."""
    def __init__(self):
        self._data = AnalogReadoutNoise()  # dummy class replaced by load()
        # loop reference for computing phase and amplitude
        self._loop = None
        # phase and amplitude data
        self._p_trace = None
        self._a_trace = None
        # noise data
        self._f_psd = None
        self._ii_psd = None
        self._qq_psd = None
        self._iq_psd = None
        self._pp_psd = None
        self._aa_psd = None
        self._pa_psd = None
        # for holding large data
        self._npz = None
        self._directory = None
        log.info("Noise object created. ID: {}".format(id(self)))

    def __getstate__(self):
        return offload_data(self, excluded_keys=("_a_trace", "_p_trace"), prefix="noise_data_")

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
    def p_trace(self):
        """
        A settable property that contains the phase trace information. Since it
        is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() is run.
        """
        if self._p_trace is None:
            raise AttributeError("The phase information has not been computed yet.")
        if isinstance(self._p_trace, str):
            return _loaded_npz_files[self._npz][self._p_trace]
        else:
            return self._p_trace

    @p_trace.setter
    def p_trace(self, phase_trace):
        self._p_trace = phase_trace

    @property
    def a_trace(self):
        """
        A settable property that contains the amplitude trace information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() is run.
        """
        if self._a_trace is None:
            raise AttributeError("The amplitude information has not been computed yet.")
        if isinstance(self._a_trace, str):
            return _loaded_npz_files[self._npz][self._a_trace]
        else:
            return self._a_trace

    @a_trace.setter
    def a_trace(self, amplitude_trace):
        self._a_trace = amplitude_trace

    @property
    def f_psd(self):
        """
        A settable property that contains the noise frequency information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._f_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._f_psd

    @f_psd.setter
    def f_psd(self, f_psd):
        self._f_psd = f_psd

    @property
    def ii_psd(self):
        """
        A settable property that contains the II noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._ii_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._ii_psd

    @ii_psd.setter
    def ii_psd(self, ii_psd):
        self._ii_psd = ii_psd

    @property
    def qq_psd(self):
        """
        A settable property that contains the QQ noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._qq_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._qq_psd

    @qq_psd.setter
    def qq_psd(self, qq_psd):
        self._qq_psd = qq_psd

    @property
    def iq_psd(self):
        """
        A settable property that contains the IQ noise information.
        Since it is derived from the i_trace and q_trace, it will raise an
        AttributeError if it is accessed before noise.compute_psd() is run.
        """
        if self._iq_psd is None:
            raise AttributeError("The IQ noise has not been computed yet.")
        return self._iq_psd

    @iq_psd.setter
    def iq_psd(self, iq_psd):
        self._iq_psd = iq_psd

    @property
    def pp_psd(self):
        """
        A settable property that contains the PP noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() and noise.compute_psd() are run.
        """
        if self._pp_psd is None:
            raise AttributeError("The phase and amplitude noise has not been computed yet.")
        return self._pp_psd

    @pp_psd.setter
    def pp_psd(self, pp_psd):
        self._pp_psd = pp_psd

    @property
    def aa_psd(self):
        """
        A settable property that contains the AA noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() and noise.compute_psd() are run.
        """
        if self._aa_psd is None:
            raise AttributeError("The phase and amplitude noise has not been computed yet.")
        return self._aa_psd

    @aa_psd.setter
    def aa_psd(self, aa_psd):
        self._aa_psd = aa_psd

    @property
    def pa_psd(self):
        """
        A settable property that contains the PA noise information.
        Since it is derived from the i_trace and q_trace and the loop, it will
        raise an AttributeError if it is accessed before
        noise.compute_phase_and_amplitude() and noise.compute_psd() are run.
        """
        if self._pa_psd is None:
            raise AttributeError("The phase and amplitude noise has not been computed yet.")
        return self._pa_psd

    @pa_psd.setter
    def pa_psd(self, pa_psd):
        self._pa_psd = pa_psd

    def clear_loop_data(self):
        """Remove all data calculated from the noise.loop attribute."""
        self.clear_traces()
        self.aa_psd = None
        self.pp_psd = None
        self.pa_psd = None

    def clear_traces(self):
        """
        Remove all trace data calculated from noise.i_trace and noise.q_trace.
        """
        self.a_trace = None
        self.p_trace = None
        self.free_memory()
        self._npz = None

    def free_memory(self, directory=None):
        """
        Offloads a_traces and p_traces to an npz file if they haven't been
        offloaded already and removes any npz file objects from memory, keeping
        just the file name. It doesn't do anything if they don't exist.
        Args:
            directory: string
                A directory string for where the data should be offloaded. The
                default is None, and the directory where the noise was saved is
                used. If it hasn't been saved, the working directory is used.
        """
        if directory is not None:
            self._set_directory(directory)
        offload_data(self, excluded_keys=("_a_trace", "_p_trace"), prefix="noise_data_")
        if isinstance(self._npz, str):
            _loaded_npz_files.free_memory(self._npz)
            _loaded_npz_files.free_memory(self._data._npz)

    def to_pickle(self, file_name):
        """Pickle and save the class as the file 'file_name'."""
        # set the _directory attributes so all the data gets saved in the right folder
        self._set_directory(os.path.dirname(os.path.abspath(file_name)))
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
        assert not isinstance(noise.f_bias, np.ndarray), "f_bias should be a number not an array"
        return noise

    def compute_phase_and_amplitude(self, label="best", fit_type="lmfit", fr=None, center=None, unwrap=True):
        """
        Compute the phase and amplitude traces stored in noise.p_trace and
        noise.a_trace.
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
                The default is None which gives the resonance frequency for the
                mkidcalculator.S21 model. This parameter determines the zero
                point for the traces.
            center: string
                An expression of parameters corresponding to the calibrated
                loop center. The default is None which gives the loop center
                for the mkidcalculator.S21 model.
            unwrap: boolean
                Determines whether or not to unwrap the phase data. The default
                is True.
        """
        compute_phase_and_amplitude(self, label=label, fit_type=fit_type, fr=fr, center=center, unwrap=unwrap)

    def compute_psd(self, **kwargs):
        """
        Compute the noise power spectral density of the noise data in this object.
        Args:
            kwargs: optional keyword arguments
                keywords for the scipy.signal.welch and scipy.signal.csd methods
        """
        # update keyword arguments
        noise_kwargs = {'nperseg': self.i_trace.shape[1], 'fs': self.sample_rate, 'return_onesided': True,
                        'detrend': 'constant', 'scaling': 'density'}
        noise_kwargs.update(kwargs)
        # compute I/Q noise in V^2 / Hz
        self.f_psd, ii_psd = welch(self.i_trace, **noise_kwargs)
        _, qq_psd = welch(self.q_trace, **noise_kwargs)
        # scipy has different order convention we use equation 5.2 from J. Gao's 2008 thesis.
        # noise_iq = F(I) conj(F(Q))
        _, iq_psd = csd(self.q_trace, self.i_trace, **noise_kwargs)
        # average multiple PSDs together
        self.ii_psd = np.mean(ii_psd, axis=0)
        self.qq_psd = np.mean(qq_psd, axis=0)
        self.iq_psd = np.mean(iq_psd, axis=0)
        try:
            # compute phase and amplitude noise in rad^2 / Hz
            _, pp_psd = welch(self.p_trace, **noise_kwargs)
            _, aa_psd = welch(self.a_trace, **noise_kwargs)
            _, pa_psd = csd(self.a_trace, self.p_trace, **noise_kwargs)
            # average multiple PSDs together
            self.pp_psd = np.mean(pp_psd, axis=0)
            self.aa_psd = np.mean(aa_psd, axis=0)
            self.pa_psd = np.mean(pa_psd, axis=0)
        except AttributeError:
            pass

    def _set_directory(self, directory):
        self._directory = directory
        try:
            self.loop._directory = self._directory
        except AttributeError:
            pass
