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
from pydub import AudioSegment

# TODO:
# 1. split each file into those segments from lines_df
# 2. need to understand how to add the label
# 3. apply FT to each file
# 4. decide how to split the test/train/validation data
# 5. research on classification model
#Note: we need to think about it - should we first train on bee and no-bee? so that we can define when we can hear bees at all?
# then we can do another model for the hive events, yes, that seems like a reasonable approach

#remove the wav ending
file_name_updated = file_name.replace('.wav','')
lines_df.reset_index(inplace=True)
to_label_df = lines_df[lines_df['file name']==file_name_updated]
to_label_df.reset_index(inplace=True)

#if we will use FT for feature engineering, then we need to split the data into bee and no bee parts so that FT is performed on all

start_time = to_label_df.loc[0,'start'] * 1000 #note: package splits in milliseconds
end_time = to_label_df.loc[0,'end'] * 1000 #note: package splits in milliseconds
bee_label = to_label_df.loc[0,'label']
inx = to_label_df.loc[0,'index']
wav = AudioSegment.from_wav('data/'+file_name)
new_wav = wav[start_time:end_time]

new_wav_name = bee_label+'_index'+str(inx)+'.wav'
new_wav.export('data/bee/'+new_wav_name, format="wav")


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
