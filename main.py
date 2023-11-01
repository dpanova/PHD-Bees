#%%
# visualize the time file

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from Annotation_data_enhancements import lines_df

#this simulates if we greb the files from the directory
file_name = 'CF003 - Active - Day - (214).wav'
s,a = wavfile.read('data/'+file_name)
print('Sampling Rate:',s) #44.1 kHZ, not an arbitrary number
print('Audio Shape:',np.shape(a)) #one channel, it looks like
#300 sec?

lenght = length = a.shape[0] / s
time = np.linspace(0., length, a.shape[0])
plt.plot(time, a)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
#%%









#%%

#adding event labels



#


#%%
#let us do FT on that split file
# we need to read the wav file with the spacy library

from numpy import fft as fft
rate,data = wavfile.read('data/bee/'+new_wav_name)
fourier=fft.fft(data)


plt.figure(1, figsize=(8,6))
plt.plot(fourier, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.show()

# the fourier is symmetrical due to the real and imaginary solution. only interested in first real solution
n = len(data)
fourier = fourier[0:int((n / 2))]

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

# calculate the frequency at each point in Hz
freqArray = np.arange(0, (n / 2), 1.0) * (rate * 1.0 / n)

plt.figure(1, figsize=(8, 6))
plt.plot(freqArray / 1000, 10 * np.log10(fourier), color='#ff7f00', linewidth=0.02)
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
plt.show()

#%%
# following the https://archive.ph/C3cw3 article
# Fourier Transform is a mathematical concept that can decompose a signal into its constituent frequencies.
import librosa
file_name = 'CF003 - Active - Day - (214).wav'
#initial stats
samples, sample_rate = librosa.load('data/'+file_name, sr=None, mono=True, offset = 0.0, duration=None)
duration_of_sound = len(samples)/sample_rate
#play the file
from IPython.display import Audio
Audio('data/'+file_name) #applicable only for the jupyter notebook
#visualize
from librosa import display
import matplotlib.pyplot as plt
plt.figure()
librosa.display.waveshow(y=samples, sr=sample_rate)
plt.xlabel('Time in sec')
plt.ylabel('Amplitude')
plt.show()
