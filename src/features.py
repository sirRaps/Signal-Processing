import numpy as np
from config import FS, WINDOW_MS, STEP_MS

def mav(window):
    """Mean Absolute Value - Average signal intensity"""

    return np.mean(np.abs(window))

def rms(window):
    """Root Mean Square - sensitive to large activations."""

    return np.sqrt(np.mean(window**2))

def zeroCrossingRate(window, threshold = 1e-6):
    """Zero Crossing Rate - related to signal frequency content."""
    zc = 0
    for i in range(1, len(window)):
        if ((window[i] >= threshold and window[i-1] < threshold) or
            (window[i] < -threshold and window[i-1] >= -threshold)):
            zc += 1

    return zc

def waveformLength(window):
    """Waveform Length — captures complexity of contraction pattern."""
    return np.sum(np.abs(np.diff(window)))

def extractFeatures(window):
    """
    Extract time-domain feature vector from a multi-channel EMG window.
    
    Args:
        window: numpy array of shape (samples, channels)
    
    Returns:
        Feature vector of length 4 * n_channels
    """
    features = []
    for ch in range(window.shape[1]):
        ch_signal = window[:, ch]
        features.extend([
            mav(ch_signal),
            rms(ch_signal),
            zeroCrossingRate(ch_signal),
            waveformLength(ch_signal)
        ])
    return np.array(features)

def extractAllWindows(emg_filtered, labels, fs=FS, 
                        window_ms=WINDOW_MS, step_ms=STEP_MS):
    """
    Extract feature matrix from full EMG recording using sliding window.
    
    Args:
        emg_filtered: filtered EMG array of shape (samples, channels)
        labels: gesture label array of shape (samples, 1)
        fs: sampling rate in Hz
        window_ms: window length in milliseconds
        step_ms: step size in milliseconds
    
    Returns:
        X: feature matrix of shape (n_windows, n_features)
        y: label array of shape (n_windows,)
    """
    window_samples = int(window_ms * fs / 1000)
    step_samples = int(step_ms * fs / 1000)
    
    X, y = [], []
    for start in range(0, len(emg_filtered) - window_samples, step_samples):
        end = start + window_samples
        window = emg_filtered[start:end, :]
        label = labels[start + window_samples//2, 0]
        X.append(extractFeatures(window))
        y.append(label)
    
    return np.array(X), np.array(y)