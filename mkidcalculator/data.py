import os
import logging
import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

_loaded_npz_files = {}  # cache of already loaded files


class AnalogReadoutABC:
    def __init__(self, npz_handle=None, index=None):
        self.index = index
        global _loaded_npz_files
        # if string load with numpy and save if it hasn't been loaded before
        if isinstance(npz_handle, str):
            npz_handle = os.path.abspath(npz_handle)
            # check if already loaded
            if npz_handle in _loaded_npz_files.keys():
                self._npz = _loaded_npz_files[npz_handle]
                log.info("loaded from cache: {}".format(npz_handle))
            else:
                npz = np.load(npz_handle)
                self._npz = npz
                log.info("loaded: {}".format(npz_handle))
                _loaded_npz_files[npz_handle] = npz
                log.info("saved to cache: {}".format(npz_handle))
        # if NpzFile skip loading but save if it hasn't been loaded before
        elif isinstance(npz_handle, np.lib.npyio.NpzFile):
            self._npz = npz_handle
            file_name = os.path.abspath(npz_handle.fid.name)
            if file_name not in _loaded_npz_files.keys():
                _loaded_npz_files[file_name] = npz_handle
                log.info("saved to cache: {}".format(file_name))
        # allow for dummy object creation
        elif npz_handle is None:
            self._npz = npz_handle
        else:
            raise ValueError("'npz_handle' must be a valid file name or a numpy npz file object.")

    def __getitem__(self, item):
        try:
            convert = self.CONVERT[item]
        except KeyError:
            raise KeyError("allowed keys for this data structure are in {}".format(list(self.CONVERT.keys())))
        try:
            result = self._npz[convert[0] if isinstance(convert, tuple) else convert]
            if result.dtype == np.dtype('O'):
                result = result.item()
            else:
                result = result[self.index]
            if isinstance(convert, tuple) and len(convert) > 1:
                result = result[convert[1]]
            return result
        except TypeError:
            raise KeyError("no data has been loaded.")


class AnalogReadoutLoop(AnalogReadoutABC):
    CONVERT = {"f": "freqs", "z": "z", "imbalance": "calibration", "offset": "z_offset", "metadata": "metadata"}


class AnalogReadoutNoise(AnalogReadoutABC):
    CONVERT = {"f_bias": "freqs", "i_trace": ("noise", "I"), "q_trace": ("noise", "Q"), "metadata": "metadata"}
    # "i_psd": ("psd", "I"), "q_psd": ("psd", "Q"), "f_psd": "f_psd" not using these from file but they are there


class AnalogReadoutPulse(AnalogReadoutABC):
    CONVERT = {"f_bias": "freqs", "i_trace": ("pulses", "I"), "q_trace": ("pulses", "Q"), "offset": "zero",
               "metadata": "metadata"}
