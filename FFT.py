# The goal of this file is to use as input the data from BeeNoBeeSplit and then apply FFT to each file

# library
from scipy.fft import fft, fftfreq
import librosa
import numpy as np
import matplotlib.pyplot as plt
# upload one file
f = 'data/bee/bee_index0.wav'
samples, sample_rate = librosa.load(f, sr=None, mono=True, offset = 0.0, duration=None)
duration_of_sound = len(samples)/sample_rate
fft(samples)

#%%
# apply hann window on top and then apply the fft
# pieces of the code are taken from here
# https://github.com/ArmDeveloperEcosystem/fixed-point-dsp-for-data-scientists/blob/main/fixed_point_dsp_for_data_scientists.ipynb

# TODO: determine the correct values here
window_size = 256
step_size = 128

number_of_windows = (len(samples) - window_size) // step_size
hanning_window = np.hanning(window_size)

# we will perform on the first window
window_1 = samples[0:window_size]
processed_window_1 = hanning_window * window_1
yf = fft(processed_window_1)




# https://pythonguides.com/python-scipy-fft/ check this out
# https://archive.ph/C3cw3
