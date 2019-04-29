import numpy as np


def compute_phase_and_amplitude(cls, label="best", fit_type="lmfit", fr="fr", center="center", unwrap=True):
    """
    Compute the phase and amplitude traces stored in pulse.phase_trace and
    pulse.amplitude_trace.
    Args:
        cls: Pulse or Noise class
            The Pulse or Noise class used to create the phase and amplitude
            data.
        label: string
            Corresponds to the label in the loop.lmfit_results or
            loop.emcee_results dictionaries where the fit parameters are.
            The resulting DataFrame is stored in
            object.loop_parameters[label]. The default is "best", which gets
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
    # get the model and parameters
    _, result_dict = cls.loop._get_model(fit_type, label)
    model = result_dict["model"]
    params = result_dict["result"].params
    # get the resonance frequency and loop center
    fr = params[fr].value
    center = params[center].value
    # get complex IQ data for the traces and loop at the resonance frequency
    traces = cls.i_trace + 1j * cls.q_trace
    z_fr = model.model(params, fr)
    f = np.empty(traces.shape)
    f.fill(cls.f_bias)
    # calibrate the IQ data
    traces = model.calibrate(params, traces, f)
    z_fr = model.calibrate(params, z_fr, fr)
    # center and rotate the IQ data
    traces = (center - traces)
    z_fr = (center - z_fr)  # should be real if no loop asymmetry
    # compute the phase and amplitude traces from the centered traces
    cls.phase_trace = np.unwrap(np.angle(traces) - np.angle(z_fr)) if unwrap else np.angle(traces) - np.angle(z_fr)
    cls.amplitude_trace = np.abs(traces) - np.abs(z_fr)
