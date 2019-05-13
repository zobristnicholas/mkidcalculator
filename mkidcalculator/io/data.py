import os
import logging
import numpy as np
from scipy.io import loadmat

from mkidcalculator.io.utils import _loaded_npz_files, offload_data, ev_nm_convert

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
                    self._energies = tuple(ev_nm_convert(laser_state))  # 1239.842 nm eV = h c
                except KeyError:
                    pass
                result = self._energies
        else:
            result = super().__getitem__(item)
        return result


def analogreadout_sweep(file_name, channel=None):
    """
    Class for loading in analogreadout sweep data.
    Args:
        file_name: string
            The sweep configuration file name.
        channel: integer
            The resonator channel for the data.
    Returns:
        loop_kwargs: list of dictionaries
            A list of keyword arguments to send to Loop.load().
    """
    directory = os.path.dirname(file_name)
    npz = np.load(file_name)
    loop_kwargs = []
    for loop_name, parameters in npz['parameter_dict'].item().items():
        loop_file_name = os.path.join(directory, loop_name)
        if os.path.isfile(loop_file_name):
            loop_kwargs.append({"loop_file_name": loop_file_name, "channel": channel})
            if parameters['noise'][0]:
                n_noise = 1 + int(parameters['noise'][5])
                noise_name = "_".join(["noise", *loop_name.split("_")[1:]])
                noise_file_name = os.path.join(directory, noise_name)
                if os.path.isfile:
                    noise_names = [noise_file_name] * n_noise
                    noise_kwargs = [{'index': ii} for ii in range(n_noise)]
                    loop_kwargs[-1].update({"noise_file_names": noise_names, "noise_kwargs": noise_kwargs,
                                            "channel": channel})
                else:
                    log.warning("Could not find '{}'".format(noise_file_name))
        else:
            log.warning("Could not find '{}'".format(loop_file_name))
    return loop_kwargs


class LegacyABC:
    def __init__(self, config_file, channel=0, index=(0, 0)):
        self.channel = channel
        self.index = index
        self._empty_fields = []
        self._do_not_clear = ['metadata']
        # load in data to the configuration file
        self._data = {'metadata': {}}
        config = loadmat(config_file, squeeze_me=True)
        for key in config.keys():
            if not key.startswith("_"):
                for name in config[key].dtype.names:
                    try:
                        self._data['metadata'][name] = float(config[name].item())
                    except ValueError:
                        self._data['metadata'][name] = config[name].item()
                    except TypeError:
                        self._data['metadata'][name] = config[name].item().astype(float)

    def __getitem__(self, item):
        value = self._data[item]
        if value is None and key not in self._empty_fields:
            self._load_data()
            value = self._data[item]
        return value

    def free_memory(self):
        """Frees memory from the wrapped data."""
        for key in self._data.keys():
            if key not in self._do_not_clear:
                self._data[key] = None

    def _load_data(self):
        raise NotImplementedError


class LegacyLoop(LegacyABC):
    def __init__(self, config_file, channel=None, index=(0, 0)):
        super().__init__(config_file, channel=channel, index=index)
        # load in the loop data
        time = os.path.basename(config_file).split('_')[2:]
        mat_file = "sweep_data.mat" if not time else "sweep_data_" + "_".join(time)
        self._mat = mat_file
        self._empty_fields += ["imbalance"]
        self._load_data()

    def _load_data(self):
        data = loadmat(self._mat, struct_as_record=False)['IQ_data']
        res = data.temps[0, self.index[0]].attens[0, self.index[1]].res[0, self.channel]
        self._data.update({"f": res.freqs, "z": res.z, "imbalance": None, "offset": res.zeropt, "field": 0,
                           "temperature": data.temprange[0, self.index[0]],
                           "attenuation": data.attenrange[0, self.index[1]]})


class LegacyNoise(LegacyABC):
    def __init__(self, config_file, channel=None, index=None):
        super().__init__(config_file, channel=channel, index=index)
        # figure out the file specifics
        directory = os.path.dirname(os.path.abspath(config_file))
        self._sweep_gui = os.path.basename(config_file).split("_")[0] == "sweep"
        if self._sweep_gui:
            if self.index is None:
                raise ValueError("The index (temperature, attenuation) must be specified for Sweep GUI data.")
            temps = np.arange(self.metadata['starttemp'], self.metadata['stoptemp'], self.metadata['steptemp'])
            attens = np.arange(self.metadata['startatten'], self.metadata['stopatten'], self.metadata['stepatten'])
            file_name = str(temps[index[0]]) + "-" + str(channel // 2 + 1) + "a-" + str(attens[index[1]]) + ".ns"
            n_points = self.metadata['adtime'] * self.metadata['noiserate'] / self.metadata['decfac']
            self._data['f_bias'] = self._data['metadata']['f0list'][self.channel]
            self._data['attenuation'] = attens[index[1]]
            self._data['sample_rate'] = self.metadata['noiserate']
        else:
            time = os.path.basename(config_file).split('.')[0].split('_')[2:]
            file_name = "pulse_data.ns" if not time else "pulse_data" + "_".join(time) + ".ns"
            n_points = self.metadata['noise_adtime'] * self.metadata['samprate']
            self._data['f_bias'] = self._data['metadata']['f0' + str(self.channel)]
            self._data['attenuation'] = self._data['metadata']['atten1'] + self._data['metadata']['atten2']
            self._data['sample_rate'] = self.metadata["samprate"]
        self._do_not_clear += ['f_bias', 'attenuation', 'sample_rate']
        # load the data
        assert n_points.is_integer(), "The noise adtime and sample rate do not give an integer number of data points"
        self._n_points = int(n_points)
        self._bin = os.path.join(directory, file_name)
        self._load_data()

    def _load_data(self):
        # get the binary data from the file
        data = np.fromfile(self._bin, dtype=np.int16)
        # remove the header from the file
        data = data[4 * 12:]
        # convert the data to voltages * 0.2 V / (2**15 - 1)
        data = data.astype(np.float16) * 0.2 / 32767.0
        # check that we have an integer number of triggers
        n_triggers = data.size / self._n_points / 4.0
        assert n_triggers.is_integer(), "non-integer number of noise traces found found in {0}".format(self._bin)
        # break noise data into I and Q data
        i_trace = np.zeros((n_triggers, self._n_points), dtype=np.float16)
        q_trace = np.zeros((n_triggers, self._n_points), dtype=np.float16)
        channel = self.channel % 2  # for if data is from sweep
        for trigger_num in range(n_triggers):
            trace_num = 4 * trigger_num
            i_trace[trigger_num, :] = data[(trace_num + 2 * channel) * self._n_points:
                                           (trace_num + 2 * channel + 1) * self._n_points]
            q_trace[trigger_num, :] = data[(trace_num + 2 * channel + 1) * self._n_points:
                                           (trace_num + 2 * channel + 2) * self._n_points]
        self._data.update({"i_trace": i_trace, "q_trace": q_trace})


class LegacyPulse(LegacyABC):
    def __init__(self, config_file, channel=None, energies=None, wavelengths=None):
        super().__init__(config_file, channel=channel)
        if energies is not None:
            self._data["energies"] = np.atleast_1d(energies)
        elif wavelengths is not None:
            self._data["energies"] = ev_nm_convert(np.atleast_1d(wavelengths))
        else:
            raise ValueError("Either energies or wavelengths must be specified.")
        self._data["f_bias"] = self.metadata["f0" + str(channel)]
        self._data["offset"] = None
        self._data["attenuation"] = self._data['metadata']['atten1'] + self._data['metadata']['atten2']
        self._do_not_clear += ['f_bias', 'attenuation', 'offset']
        self._empty_fields += ["offset"]

        # TODO: self._bin =
        #  self._n_points =

    def _load_data(self):
        pass # TODO: write this method


def legacy_sweep(config_file):
    pass
