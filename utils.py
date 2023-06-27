import os
import librosa

def load_pink():
    ap_pink_bl_1 = os.path.join("grabaciones","posicion 1","pinknoise position BACK L.wav") 
    ap_pink_br_1 = os.path.join("grabaciones","posicion 1","pinknoise position BACK R.wav")
    ap_pink_bl_2 = os.path.join("grabaciones","posicion 2","pinknoise position BACK L.wav") 
    ap_pink_br_2 = os.path.join("grabaciones","posicion 2","pinknoise position BACK R.wav")

    pink_bl_1 , sr = librosa.load(ap_pink_bl_1, sr=None)
    pink_br_1 , sr = librosa.load(ap_pink_br_1, sr=None)
    pink_bl_2 , sr = librosa.load(ap_pink_bl_2, sr=None)
    pink_br_2 , sr = librosa.load(ap_pink_br_2, sr=None)

    return pink_bl_1, pink_br_1, pink_bl_2, pink_br_2, sr
def load_floor():
    ap_floor_bl_1 = os.path.join("grabaciones","posicion 1","background position BACK L.wav") 
    ap_floor_br_1 = os.path.join("grabaciones","posicion 1","background position BACK R.wav")
    ap_floor_bl_2 = os.path.join("grabaciones","posicion 2","background position BACK L.wav") 
    ap_floor_br_2 = os.path.join("grabaciones","posicion 2","background position BACK R.wav")

    floor_bl_1 , sr = librosa.load(ap_floor_bl_1, sr=None)
    floor_br_1 , sr = librosa.load(ap_floor_br_1, sr=None)
    floor_bl_2 , sr = librosa.load(ap_floor_bl_2, sr=None)
    floor_br_2 , sr = librosa.load(ap_floor_br_2, sr=None)

    return floor_bl_1, floor_br_1, floor_bl_2, floor_br_2, sr

def load_calibration():

    ap_calibration_1 = os.path.join("grabaciones","posicion 1","puretone position FRONT.wav")
    ap_calibration_2 = os.path.join("grabaciones","posicion 1","puretone position BACK R.wav")

    audio_calibration_1 ,sr = librosa.load(ap_calibration_1, sr=None)
    audio_calibration_2 ,sr = librosa.load(ap_calibration_2, sr=None)

    return audio_calibration_1, audio_calibration_2, sr

def load_sweep():

    # Audio paths. Position 1
    ap_bl_1 = os.path.join("grabaciones","posicion 1","sinesweep position BACK L.wav")
    ap_br_1 = os.path.join("grabaciones","posicion 1","sinesweep position BACK R.wav") 
    ap_f_1 = os.path.join("grabaciones","posicion 1","sinesweep position FRONT.wav") 

    # Position 2
    ap_bl_2 = os.path.join("grabaciones","posicion 2","sinesweep position BACK L.wav") 
    ap_br_2 = os.path.join("grabaciones","posicion 2","sinesweep position BACK R.wav") 
    ap_f_2 = os.path.join("grabaciones","posicion 2","sinesweep position FRONT.wav")

    # Anechoic audio
    ap_audio_ane = os.path.join("audios","anechoic_voice.wav")

    # Load the data. Positon 1

    sweep_bl_1, sr = librosa.load(ap_bl_1, sr=None)
    sweep_br_1, sr = librosa.load(ap_br_1, sr=None)
    sweep_f_1, sr = librosa.load(ap_f_1, sr=None)

    # Load the data. Positon 2

    sweep_bl_2, sr = librosa.load(ap_bl_2, sr=None)
    sweep_br_2, sr = librosa.load(ap_br_2, sr=None)
    sweep_f_2, sr = librosa.load(ap_f_2, sr=None)

    # load Anechoich voice
    audio_ane, sr = librosa.load(ap_audio_ane, sr=48000)

    return sweep_bl_1, sweep_br_1, sweep_f_1, sweep_bl_2, sweep_br_2, sweep_f_2, audio_ane, sr
