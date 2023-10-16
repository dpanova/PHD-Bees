# The goal of this file is to use as input the data from BeeNoBeeSplit and then apply FFT to each file

# library
from scipy.fft import fft, fftfreq
import librosa

# upload one file
f = 'data/bee/bee_index0.wav'
samples, sample_rate = librosa.load(f, sr=None, mono=True, offset = 0.0, duration=None)
duration_of_sound = len(samples)/sample_rate

fft(samples)
# https://pythonguides.com/python-scipy-fft/ check this out
# https://archive.ph/C3cw3
