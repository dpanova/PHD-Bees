#%%
# visualize the time file

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


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
import pandas as pd
import math

with open('beeAnnotations.mlf') as f:
    lines = f.readlines()


lines_df = pd.DataFrame(lines)

lines_df['all']=lines_df[0].apply(lambda x:x.split('\t'))

def split_list(row, column_name):
    return pd.Series(row[column_name])

lines_df=lines_df.apply(lambda x:split_list(x,'all'), axis=1)

lines_df.rename(columns={0:'start',1:'end',2:'label'},inplace=True)


# then remove the /n rows and null rows

#lines_df['end'].apply(lambda x: if isnull()]

def file_name_extract(row, column_name1, column_name2):
    if pd.isnull(row[column_name2]):
        label = row[column_name1]
    else:
        label = math.nan
    return label

lines_df['file name'] = lines_df.apply(lambda x: file_name_extract(x, 'start','end'), axis=1)
lines_df['file name'].ffill(inplace=True)
#remove empty rows
lines_df = lines_df[(lines_df['file name'] != '.\n') & (~lines_df['end'].isna())]

#remove new line character
lines_df['file name'] = lines_df['file name'].str.replace('\n','')
lines_df['label'] = lines_df['label'].str.replace('\n','')

#change data types
lines_df['start'] = lines_df['start'].astype(float)
lines_df['end'] = lines_df['end'].astype(float)

#%%

#adding event labels

#adding event labels
lines_df['missing queen'] = lines_df['file name'].str.lower().str.contains('|'.join(['missing queen', 'no_queen']))
lines_df['queen'] = (lines_df['file name'].str.lower().str.contains('queenbee')) & (~lines_df['file name'].str.lower().str.contains('no'))
lines_df['active day'] = lines_df['file name'].str.lower().str.contains('active - day')
lines_df['swarming'] = lines_df['file name'].str.lower().str.contains('swarming')

#distirbution across files
lines_df[['missing queen','queen','active day','swarming','file name']].groupby(['missing queen','queen','active day','swarming'],as_index=False).nunique()
#data is balanced for missing queen and queen, we have 15 active days files and only 2 swarming, we don't have inactive days data

#let us see in terms of overall seconds since files do not have equal distribution
lines_df['duration'] = lines_df['end'] - lines_df['start']
lines_df[['missing queen','queen','active day','swarming','duration']].groupby(['missing queen','queen','active day','swarming'],as_index=False).sum()
#in terms of hours of recording no qeen and queen are somewhat balanced


#maybe let us see what is the distribution in terms of bee and nobee
lines_df[['label','duration']].groupby('label',as_index=False).sum()
#not balanced at all

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

start_time = to_label_df.loc[0,'start'] * 1000 #note: package splits in miliseconds
end_time = to_label_df.loc[0,'end'] * 1000 #note: package splits in miliseconds
bee_label = to_label_df.loc[0,'label']
inx = to_label_df.loc[0,'index']
wav = AudioSegment.from_wav('data/'+file_name)
new_wav = wav[start_time:end_time]

new_wav_name = bee_label+'_index'+str(inx)+'.wav'
new_wav.export('data/bee/'+new_wav_name, format="wav")


#%%
#let us do FT on that splitted file
# we need to read the wav file with the spacy library

from numpy import fft as fft
rate,data = wavfile.read('data/bee/'+new_wav_name)
fourier=fft.fft(data)


plt.figure(1, figsize=(8,6))
plt.plot(fourier, color='#ff7f00')
plt.xlabel('k')
plt.ylabel('Amplitude')
plt.show()

# the fourier is symetrical due to the real and imaginary soultion. only interested in first real solution
n = len(data)
fourier = fourier[0:int((n / 2))]

# scale by the number of points so that the magnitude does not depend on the length
fourier = fourier / float(n)

# calculate the frequency at each point in Hz
freqArray = np.arange(0, (n / 2), 1.0) * (rate * 1.0 / n);

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
