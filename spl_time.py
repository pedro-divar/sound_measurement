# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 16:16:20 2023

@author: marco
"""
import numpy as np

def spl_time(sr_audio, audio):
    """
    Calculate the Sound Pressure Level (SPL) over time of an audio signal relative to a calibration signal.

    Parameters:
        audio_path (str): File path to the audio signal WAV file.
        calibration_path (str): File path to the calibration signal WAV file.

    Returns:
        tuple: A tuple containing two arrays.
            - t (numpy.ndarray): Time values in seconds corresponding to the audio signal.
            - audio_dB (numpy.ndarray): Sound Pressure Level (SPL) values in decibels (dB) relative to 20 µPa.
    Notes:
        The calibration signal WAV file should have an RMS value of 1 Pascal (94 dB SPL).

    """

    audio_dB = 20 * np.log10(abs(audio / 20e-6)+0.0000000000001)  # Convert audio signal to decibels (dB) relative to 20 µPa

    t = np.linspace(0, len(audio) / sr_audio, len(audio))  # Time values corresponding to the audio signal

    return t, audio_dB

def calibration(audio, calib):
    
    calib_rms = np.sqrt(np.mean(calib**2))  # Root Mean Square (RMS) of calibration signal in Pascal
    return audio / calib_rms 