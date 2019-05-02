import os
import logging
import numpy as np

from mkidcalculator.io.utils import _loaded_npz_files, offload_data

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def analogreadout_temperature(metadata):
    """
    Returns the average temperature across all the temperatures taken during
    the data set.
    Args:
        metadata: dictionary
            The metadata dictionary from the analogreadout procedure.
    Returns:
        temperature: float
            The average temperature during the data set.
    """
    temperatures = [metadata[key]['thermometer']['temperature'][0]
                    for key in metadata.keys() if key not in ('parameters', 'file_name')]
    temperature = np.mean(temperatures)
    return temperature


def analogreadout_sample_rate(metadata):
    """
    Returns the sample rate in Hz from the metadata.
    Args:
        metadata: dictionary
            The metadata dictionary from the analogreadout procedure.
    Returns:
        sample_rate: float
            The sample rate in Hz.
    """
    sample_rate = metadata['parameters']['sample_rate'] * 1e6
    return sample_rate


class AnalogReadoutABC:
    """
    Abstract base class for handling data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    def __init__(self, npz_handle=None, channel=None, index=None):
        self.channel = channel
        self.index = index
        npz = _loaded_npz_files[npz_handle]  # caches file
        self._npz = None if npz is None else os.path.abspath(npz.fid.name)

    def __getstate__(self):
        return offload_data(self)

    def __getitem__(self, item):
        # get conversion values
        try:
            convert = self.CONVERT[item]
        except KeyError:
            raise KeyError("allowed keys for this data structure are in {}".format(list(self.CONVERT.keys())))
        try:
            # get the result from the npz file
            result = _loaded_npz_files[self._npz][convert[0] if isinstance(convert, tuple) else convert]
            if result.dtype == np.dtype('O'):
                # if it's an object unpack it
                result = result.item()
            else:
                # else get the channel and index
                if self.index is not None:
                    result = result[:, self.index, ...]
                if self.channel is not None:
                    result = result[self.channel]
            # more conversion
            if isinstance(convert, tuple) and len(convert) > 1:
                # try the first conversion
                if not callable(convert[1]):
                    try:
                        if isinstance(convert[1], (tuple, list)):
                            for c in convert[1]:
                                result = result[c]
                        else:
                            result = result[convert[1]]
                    except IndexError:  # didn't work try the second
                        # if that failed run a function (for complex formatted data)
                        result = convert[2](result)
                else:
                    result = convert[1](result)
            return result
        except TypeError:
            raise KeyError("no data has been loaded.")

    def free_memory(self):
        """Frees memory from the wrapped npz file."""
        _loaded_npz_files.free_memory(self._npz)


class AnalogReadoutLoop(AnalogReadoutABC):
    """
    Class for handling loop data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    CONVERT = {"f": "freqs", "z": "z", "imbalance": "calibration", "offset": "z_offset", "metadata": "metadata",
               "attenuation": ("metadata", ("parameters", "attenuation")),
               "field": ("metadata", ("parameters", "field")), "temperature": ("metadata", analogreadout_temperature)}


class AnalogReadoutNoise(AnalogReadoutABC):
    """
    Class for handling noise data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    CONVERT = {"f_bias": "freqs", "i_trace": ("noise", "I", np.real), "q_trace": ("noise", "Q", np.imag),
               "metadata": "metadata", "attenuation": ("metadata", ("parameters", "attenuation")),
               "sample_rate": ("metadata", analogreadout_sample_rate)}
    # "i_psd": ("psd", "I"), "q_psd": ("psd", "Q"), "f_psd": "f_psd" not using these from file but they are there


class AnalogReadoutPulse(AnalogReadoutABC):
    """
    Class for handling pulse data from the analogreadout module.
    Args:
        npz_handle: string or numpy.lib.npyio.NpzFile object (optional)
            A file path or numpy npz file object containing the data. The
            default is None, which creates a dummy object with no data that
            raises a KeyError when accessed.
        channel: integer (optional)
            An integer specifying which channel to load. The default is None
            and all channels are returned.
        index: integer (optional)
            An integer specifying which index to load. The default is None and
            all indices will be returned.
    """
    CONVERT = {"f_bias": "freqs", "i_trace": ("pulses", "I", np.real), "q_trace": ("pulses", "Q", np.imag),
               "offset": "zero", "metadata": "metadata", "attenuation": ("metadata", ("parameters", "attenuation")),
               "sample_rate": ("metadata", analogreadout_sample_rate)}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._energies = []

    def __getitem__(self, item):
        if item == 'energies':
            if self._energies:
                result = self._energies
            else:
                metadata = super().__getitem__("metadata")
                try:
                    laser_state = np.array(metadata['parameters']['laser'])
                    laser_state *= np.array([808, 920, 980, 1120, 1310])
                    laser_state = laser_state[laser_state != 0]
                    self._energies = tuple(1239.842 / laser_state)  # 1239.842 nm eV = h c
                except KeyError:
                    pass
                result = self._energies
        else:
            result = super().__getitem__(item)
        return result


def analogreadout_sweep(file_name):
    directory = os.path.dirname(file_name)
    npz = np.load(file_name)
    loop_kwargs = []
    for loop_name, parameters in npz['parameter_dict'].item().items():
        loop_file_name = os.path.join(directory, loop_name)
        if os.path.isfile(loop_file_name):
            loop_kwargs.append({"loop_file_name": loop_file_name})
            if parameters['noise'][0]:
                n_noise = 1 + int(parameters['noise'][5])
                noise_name = "_".join(["noise", *loop_name.split("_")[1:]])
                noise_file_name = os.path.join(directory, noise_name)
                if os.path.isfile:
                    noise_names = [noise_file_name] * n_noise
                    noise_kwargs = [{'index': ii} for ii in range(n_noise)]
                    loop_kwargs[-1].update({"noise_file_names": noise_names, "noise_kwargs": noise_kwargs})
                else:
                    log.warning("Could not find '{}'".format(noise_file_name))
        else:
            log.warning("Could not find '{}'".format(loop_file_name))
    return loop_kwargs
