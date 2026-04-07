import numpy as np
from scipy.signal import butter, filtfilt
from config import FS, LOWCUT, HIGHCUT, FILTER_ORDER

def bandpass_filter(signal, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=FILTER_ORDER):
    """
    Apply zero-phase Butterworth bandpass filter to EMG signal.
    
    Args:
        signal: 1D numpy array of raw EMG samples
        lowcut: lower frequency bound in Hz (default 20)
        highcut: upper frequency bound in Hz (default 500)
        fs: sampling rate in Hz (default 2000)
        order: filter order (default 4)
    
    Returns:
        Filtered signal as 1D numpy array
    """
    
    nyq = fs / 2
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    
    return filtfilt(b, a, signal)

def filter_all_channels(emg):
    """
    Apply bandpass filter to all channels of an EMG matrix.
    
    Args:
        emg: numpy array of shape (samples, channels)
    
    Returns:
        Filtered EMG matrix of same shape
    """
    
    emg_filtered = np.zeros_like(emg)

    for ch in range(emg.shape[1]):
        emg_filtered[:, ch] = bandpass_filter(emg[:, ch])

    return emg_filtered

def get_envelope(signal, window_ms=200, fs=FS):
    """
    Compute EMG envelope using sliding RMS window.
    
    Args:
        signal: 1D numpy array of filtered EMG samples
        window_ms: RMS window length in milliseconds
        fs: sampling rate in Hz
    
    Returns:
        Envelope signal as 1D numpy array
    """

    window_samples = int(window_ms * fs / 1000)
    envelope = np.array([
        np.sqrt(np.mean(signal[i:i + window_samples]**2))
        for i in range(len(signal) - window_samples)
    ])

    return envelope




