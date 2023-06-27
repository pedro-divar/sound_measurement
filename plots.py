import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

# plot filter in magnitud and phase
def plot_filter_response(b, a, fs,freqc):
    freq = np.linspace(0, 0.5 * fs, 10000)  # Generate a frequency vector

    w, h = freqz(b, a, worN=freq, fs=fs)  # Compute the frequency response

    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(freq, 20 * np.log10(abs(h)))  # Plot the magnitude response in dB
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'Frequency Response - Magnitude {freqc} Hz')
    plt.ylim(-60, 5)  # Set the y-axis limits
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freq, np.angle(h))  # Plot the phase response in radians
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    plt.title(f'Frequency Response - Phase {freqc} Hz')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_fft_signal(signal, fs, title, ticks, xlim=(50, 22000), ylim=(-60, 5)):
    """
    Plots the magnitude spectrum of a signal using FFT.
    
    Args:
        signal (ndarray): input signal.
        fs (float): sampling rate of the signal.

    Returns:
        None (displays a plot).
    """
    
    # Compute the FFT of the signal
    fft_raw = np.fft.fft(signal)
    
    # Keep only the positive frequencies
    fft = fft_raw[:len(fft_raw)//2]
    
    # Compute the magnitude of the FFT
    fft_mag = abs(fft) / len(fft)
    
    # Compute the corresponding frequencies
    freqs = np.linspace(0, fs/2, len(fft), endpoint=False)

    # Normalize the magnitude to the maximum value
    fft_mag_norm = fft_mag / np.max(abs(fft_mag))
    
    # Add a small epsilon to avoid numerical issues when computing the logarithm
    eps = np.finfo(float).eps
    
    # Convert the magnitude to decibels (dB)
    fft_mag_db = 20*np.log10(fft_mag_norm + eps)

    # Plot the magnitude spectrum in dB
    plt.figure(figsize=(15,5))
    plt.semilogx(freqs, fft_mag_db)
    plt.title(title)
    plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid()
    plt.show()

def plot_signal(t, signal, title):

    plt.plot(t, signal)
    plt.xlabel("time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()

def plot_dB(t,signal, title, color):
    plt.plot(t, 20*np.log10(abs(signal)),alpha=0.7,color=color)
    plt.ylim(-70,0)
    plt.xlim(0,0.8)
    ticks = [0,-20,-30,-60]
    plt.yticks([t for t in ticks], [f'{t}' for t in ticks])
    plt.title(title)
    plt.xlabel("Timse[s]")
    plt.ylabel("dB normalized")
    plt.grid()
    plt.show()
