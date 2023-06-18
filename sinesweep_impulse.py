import numpy as np
from numpy.fft import rfft, irfft
import soundfile as sf
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
import os

def sinesweep (f1, f2, T, fs):

    t = np.linspace(0, T, int(fs * T) )
    R = np.log(f2 / f1)

    # ESS generation
    sinesweep = np.sin((2 * np.pi * f1 * T / R) * (np.exp(t * R / T) - 1))
    return sinesweep / np.max(np.abs(sinesweep))

def inverse_sinesweep(f1, f2, T, fs):

    t = np.linspace(0, T, int(fs * T) )
    R = np.log(f2 / f1)   

    sinesweep_array = sinesweep(f1, f2, T, fs)
    k = np.exp((t * R) / T)
    inverse_sinesweep = sinesweep_array[::-1] / k

    return inverse_sinesweep / np.max(np.abs(inverse_sinesweep))

def impulse(inverse_sinesweep, record_sinesweep):

    impulse = fftconvolve(inverse_sinesweep, record_sinesweep, mode = "same")
    impulse = impulse / np.max(np.abs(impulse))

    return impulse

def pink(T, fs) -> np.ndarray:
    """
    Pink noise.
    * T: duration of the noise.
    Pink noise has equal power in bands that are proportionally wide.
    Power density decreases with 3 dB per octave.
    https://github.com/python-acoustics
    """
    N = T*fs

    # This method uses the filter with the following coefficients.
    # b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    # a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
    # return lfilter(B, A, np.random.randn(N))

    # Another way would be using the FFT


    white = np.random.randn(N).astype(np.float32)
    fft_white = rfft(white) / N

    S = np.sqrt(np.arange(fft_white.size) + 1.0)  # +1 to avoid divide by zero
    pink = irfft(fft_white / S).real[:N]

    return pink / np.max(np.abs(pink))  # extremely tiny value 1e-9 without normalization


def write_audio(fs,audio,title ="audio"):

    # Create directory if it doesn't exist
    os.makedirs("arreglos", exist_ok=True)
    # Join the directory and file name using os.path.join()
    file_path = os.path.join("arreglos", f"{title}_{fs}.npy")
    np.save(file_path, audio)

    # Create directory if it doesn't exist
    os.makedirs("audios", exist_ok=True)
    # Join the directory and file name using os.path.join()
    file_path = os.path.join("audios", f"{title}.wav")
    sf.write(file_path, audio, fs)

if __name__=="__main__":

    fs = 44100
    f1 = 100
    f2 = 16000
    T_sweep = 30
    T_pink = 30

    sweep = sinesweep(f1, f2, T_sweep, fs)
    pink_noise = pink(T_pink, fs)
    inverse_sweep = inverse_sinesweep(f1, f2, T_sweep, fs)

    write_audio(fs, sweep, "sinesweep")
    write_audio(fs, pink_noise, "pink_noise")

    # Test recorded sinesweep
    record_sinesweep, fs = sf.read("sinesweep.wav")
    impulse_test = impulse(inverse_sweep, record_sinesweep)


    plt.plot(record_sinesweep)
    plt.show()

    
    


