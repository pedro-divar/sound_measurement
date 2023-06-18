from scipy.signal import butter, freqz, lfilter
import matplotlib.pyplot as plt
import numpy as np


def butter_filter(f_type, fs, order, cutoff_freq):
    nyquist_freq = 0.5 * fs

    if f_type == 'lowpass':
        cutoff_norm = cutoff_freq/nyquist_freq
        b, a = butter(order, cutoff_norm, btype='lowpass')  # Generate coefficients for a lowpass Butterworth filter
    elif f_type == 'highpass':
        cutoff_norm = cutoff_freq/nyquist_freq
        b, a = butter(order, cutoff_norm, btype='highpass')  # Generate coefficients for a highpass Butterworth filter
    elif f_type == 'bandpass':
        cutoff_norm = [freq / (nyquist_freq) for freq in cutoff_freq]
        low_freq = cutoff_freq[0]
        high_freq = cutoff_freq[1]
        low_norm = low_freq / nyquist_freq
        high_norm = high_freq / nyquist_freq
        b, a = butter(order, [low_norm, high_norm], btype='bandpass')  # Generate coefficients for a bandpass Butterworth filter
    elif f_type == 'bandstop':
        cutoff_norm = [freq / (nyquist_freq) for freq in cutoff_freq]
        low_freq = cutoff_freq[0]
        high_freq = cutoff_freq[1]
        low_norm = low_freq / nyquist_freq
        high_norm = high_freq / nyquist_freq
        b, a = butter(order, [low_norm, high_norm], btype='bandstop')  # Generate coefficients for a bandstop Butterworth filter
    else:
        raise ValueError("Invalid filter")  # Raise an error for an invalid filter type

    return b, a

# plot filter in magnitud and phase
def plot_filter_response(b, a, fs):
    freq = np.linspace(0, 0.5 * fs, 10000)  # Generate a frequency vector

    w, h = freqz(b, a, worN=freq, fs=fs)  # Compute the frequency response

    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(freq, 20 * np.log10(abs(h)))  # Plot the magnitude response in dB
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Frequency Response - Magnitude')
    plt.ylim(-60, 5)  # Set the y-axis limits
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freq, np.angle(h))  # Plot the phase response in radians
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    plt.title('Frequency Response - Phase')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def apply_filter(b, a, signal):
    return lfilter(b, a, signal)