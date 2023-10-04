# This is a sample Python script.

#This file is to explore how to transform wav file to FT


import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import scipy.io.wavfile as wav


sample_rate, data = wav.read('data/CF003 - Active - Day - (214).wav')
#sample_rate, data = wav.read('audio.wav')

import numpy as np
from scipy.fft import fft

# Convert audio data to frequency domain
fft_data = fft(data)

# Calculate the corresponding frequencies
frequencies = np.fft.fftfreq(len(data), 1 / sample_rate)

# Print the dominant frequencies
# for i, freq in enumerate(frequencies):
#     if np.abs(fft_data[i]) > threshold:
#         print(freq, np.abs(fft_data[i]))

#%%
# visualize the time file

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment

s,a = wavfile.read('data/CF003 - Active - Day - (214).wav')
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
# TO DO:
# 1. split each file into those segments from lines_df
# 2. need to understand how to add the label
# 3. apply FT to each file
# 4. decide how to split the test/train/validation data
# 5. research on classification model

f =
