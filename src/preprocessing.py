import numpy as np
from scipy.signal import butter, filtfilt
from config import FS, LOWCUT, HIGHCUT, FILTER_ORDER

def bandpassFilter(signal, lowcut=LOWCUT, highcut=HIGHCUT, fs=FS, order=FILTER_ORDER):
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

def filterAllChannels(emg):
    """
    Apply bandpass filter to all channels of an EMG matrix.
    
    Args:
        emg: numpy array of shape (samples, channels)
    
    Returns:
        Filtered EMG matrix of same shape
    """
    
    emgFiltered = np.zeros_like(emg)

    for ch in range(emg.shape[1]):
        emgFiltered[:, ch] = bandpassFilter(emg[:, ch])

    return emgFiltered

def getEnvelope(signal, windowMs = 200, fs = FS):
    """
    Compute EMG envelope using sliding RMS window.
    
    Args:
        signal: 1D numpy array of filtered EMG samples
        window_ms: RMS window length in milliseconds
        fs: sampling rate in Hz
    
    Returns:
        Envelope signal as 1D numpy array
    """

    windowSamples = int(windowMs * fs / 1000)
    envelope = np.array([
        np.sqrt(np.mean(signal[i:i+windowSamples]**2))
        for i in range(len(signal) - windowSamples)
    ])

    return envelope




