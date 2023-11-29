# The goal of this file is the following:
# 1. Split the data into train and test, y-label -> bee or not to bee
# 2. extract FFT and CWT coefficients (continuous wavelet transformation)
# 3. Compare the following models:
# - Random FOrest
# - SVM
# - XGBoost
# - Light GBM
# - Deep Neural Network

# library
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from scipy.fft import fft, rfftfreq
import librosa
import numpy as np
#%%
# Test-train
# Do we actually need to stratify?
annotation_df = pd.read_csv('beeAnnotations_enhanced.csv')
annotation_df[['label', 'duration']].groupby('label', as_index=False).sum()
annotation_df[['label', 'duration']].groupby('label', as_index=False).count()

# No we don't need to stratify the split. We are not interested in the duration of the file but rather the frequencies,
# since we are moving away from the time domain and we go for the frequency domain, then we will look into the count of
# observations which is balanced for this label

# to perform the random split we will use the annotation file
X_train_index, X_test_index, y_train, y_test = train_test_split(annotation_df[['index']], annotation_df[['label']],
                                                    test_size=0.25)

# save all files for reproducibility
X_train_index.to_csv('X_train_index.csv')
X_test_index.to_csv('X_test_index.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')
#%%
# get a list of all split files
bee_folder = 'data/bee/'
no_bee_folder = 'data/nobee/'

bee_files = os.listdir(bee_folder)
nobee_files = os.listdir(no_bee_folder)

# DF to store the results
X_train = pd.DataFrame()

#%%
len(X_train_index)
X_train_part1 = X_train_index.iloc[:10,:]




#%%
# transform the train set with fft and store it to a DF
for train_index,row in X_train_part1.iterrows():
    print(train_index)
    # get the necessary indices
    file_index = row['index'] # this index is necessary to ensure we have the correct file name
    label = y_train.loc[train_index,'label']
    # locate the correct file
    print('3')
    try:
        if label =='bee':
            file_name = [x for x in bee_files if x.find('index'+str(file_index)+'.wav') != -1][0]
            file_name = bee_folder+file_name
        else:
            file_name = [x for x in nobee_files if x.find('index' + str(file_index) + '.wav') != -1][0]
            file_name = nobee_files+file_name
    except:
        print('no such file')
    print('2')
    # try to read the file
    try:
        samples, sample_rate = librosa.load(file_name, sr=None, mono=True, offset = 0.0, duration=None)
        #check if the data extraction is correct
        duration_of_sound = len(samples)/sample_rate
        annotation_duration = annotation_df.loc[annotation_df['index']==file_index, 'duration']

        # we need to do different error handling with log files at a later point
        if duration_of_sound==annotation_duration.to_list()[0]:
            print('file is read correctly')
        else:
            print('file is not read correctly')

        # calculate the fft
        # we need to get only the real part of FFT -> need to check on this
        print('1')
        fft_file = fft(samples).tolist()
        fft_file_real = [x.real for x in fft_file]

        # we need to add somewhere the train index
        fft_file_real.append(train_index) # question: can this be a different length?
        fft_file_real.append(file_index)

        X_train = X_train._append(pd.DataFrame([fft_file_real]))
    except:
        print('lab exception file')
X_train.to_csv('X_train.csv', index=False)
#%%
# it looks like only 53 out of 100 have been transformed
# we need to change the file to get the index as the first argument
# we need to check why fft does not provide the same len vector
# all of them are not read correctly - why?

# different sampling rates - different fft vector lengths

# we will investigate the two usecases
train_index1 = 0
train_index2 = 7
# get the necessary indices

file_index1 = X_train_index.loc[train_index1,] # this index is necessary to ensure we have the correct file name
file_index2 = X_train_index.loc[train_index2,]

label1 = y_train.loc[train_index1,'label']
label2 = y_train.loc[train_index2,'label']
# locate the correct file


file_name1 = [x for x in bee_files if x.find('index'+str(file_index1[0])+'.wav') != -1][0]
file_name1 = bee_folder+file_name1
file_name2 = [x for x in bee_files if x.find('index' + str(file_index2[0]) + '.wav') != -1][0]
file_name2 = bee_folder+file_name2


# file 1
samples1, sample_rate1 = librosa.load(file_name1, sr=None, mono=True, offset = 0.0, duration=None)
len(samples1) #496125
sample_rate1 #44100
duration_of_sound1 = len(samples1)/sample_rate1 #11.25
annotation_duration1 = annotation_df.loc[annotation_df['index']==file_index1[0], :]['duration'][train_index1] # this could be the issue itself, it was substracting the whole table

duration_of_sound==annotation_duration


fft_file1 = fft(samples1).tolist()
fft_file_real1 = [x.real for x in fft_file1]
len(fft_file_real1) #496125, the same number as the samples

# let us do the same for sample 2

samples2, sample_rate2 = librosa.load(file_name2, sr=None, mono=True, offset = 0.0, duration=None)
len(samples2) #220500
sample_rate2 #44100
duration_of_sound2 = len(samples2)/sample_rate2 #5.0
annotation_duration2 = annotation_df.loc[annotation_df['index']==file_index2[0], :]['duration'][train_index2]
# this could be the issue itself, it was substracting the whole table
# also the index is to be highlighted

duration_of_sound==annotation_duration


fft_file2 = fft(samples2).tolist()
fft_file_real2 = [x.real for x in fft_file2]
len(fft_file_real2) #220500, the same number as the samples

N1 = len(samples1)
fft_frq1 = rfftfreq(N1, d = 1.0/sample_rate1)
len(fft_frq1) #248063

N2 = len(samples2)
fft_frq2 = rfftfreq(N2, d = 1.0/sample_rate2)
len(fft_frq2) #248063

# what if we apply window length and calculate the fft? so that we have the same length of the array?

def dht(x: np.array):
    """ Compute the DHT for a sequence x of length n using the FFT.
    """
    X = np.fft.fft(x)
    X = np.real(X) - np.imag(X)
    return X

# TODO
# a = fft(samples1*np.hanning(len(samples1))) #adding hann window function

a = dht(samples1)
freqs = np.fft.fftfreq(len(samples1), sample_rate1)

# we can split the data into 128 bins
#
# The output from the FHT is a series of frequency ‘bins’, the value of each bin represents the intensity of the input signal within the range of frequencies the bin represents.



# # need to understand the difference between FFT and fast hartley transformation
# import ducc0
# from ducc0 import  fft as fftd
# fftd()

#%%
# following the example in this notebook https://raghavchhetri.github.io/scattered.dimes/2021/07/21/Fourier-Transforms-in-Python
from scipy.fft import fft, ifft, fftn, ifftn, fftshift, ifftshift, fftfreq, rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
X = fft(samples1)
freq = fftfreq(len(X),(1/sample_rate1))
powerspect = 2*np.abs(X)/len(X)
# denoice - cutoff all frequencies with small power
cutoff = 0
powerspect = powerspect * (powerspect > cutoff)
X = X * (powerspect > cutoff) # Zero small Fourier coefficients
t = np.arange(0,duration_of_sound1,(1/sample_rate1))

plt.figure(figsize=(9, 7))
plt.subplot(311)
plt.plot(t, samples1, 'k', label='original')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(312)
plt.title('frq')
# plt.stem(freq, np.abs(X),'c', markerfmt=" ", basefmt="-b")
plt.stem(freq, powerspect, 'c', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude')
plt.xlim(-2000, 2000)
plt.subplot(313)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

#%% define the function in a separate piece of code

def fft_powerspectrum_plot(x, t, dt, npnts, freq_show, method=2, denoise=False, cutoff=0):
    '''
    - plot original signal
    - compute FFT and power spectrum (power per frequency)
    - plot power spectrum
    - plot recovered signal after fft-> ifft

    x:         1D signal
    t:         time vector
    dt:        sampling interval
    npnts:     number of time points
    freq_show: upper range of frequency to plot
    method:    1/2/3
    denoise:   filter out noise
    cutoff:    denoise below the cutoff amplitude

    Call: plot_signal_fftamplitude(x, t, dt, npnts, 200, method=1, denoise=True, cutoff=3)
    '''
    if method == 1:
        X = fft(x)
        n = np.arange(npnts)
        T = npnts * dt
        freq = n / T
        title = 'Frequency mirroring above Nyquist'
    elif method == 2:
        X = fft(x)
        freq = fftfreq(npnts, dt)
        title = 'Frequency mirroring about zero'
    elif method == 3:
        X = rfft(x)
        freq = rfftfreq(npnts, dt)
        title = 'No mirroring: only positive frequencies'

    powerspect = 2 * np.abs(X) / npnts
    # Note: Normalized as 2*np.abs(X)/npnts instead of simply np.abs(X)
    # returns the actual amplitude values of the sine and cosine functions
    # instead of some arbitrarily-scaled values

    if denoise:
        powerspect = powerspect * (powerspect > cutoff)  # Zero all frequencies with small power
        X = X * (powerspect > cutoff)  # Zero small Fourier coefficients
        # To further zero out a peak at zero frequency -- occurs if noise is 'rand' instead of 'randn'
        # powerspect = powerspect * (powerspect < 10)
        # X = X * (powerspect < 10)

    # PLOT
    plt.figure(figsize=(9, 7))
    plt.subplot(311)
    plt.plot(t, x, 'k', label='original')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(312)
    plt.title(title)
    # plt.stem(freq, np.abs(X),'c', markerfmt=" ", basefmt="-b")
    plt.stem(freq, powerspect, 'c', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude')
    if method == 2:
        plt.xlim(-freq_show, freq_show)
    else:
        plt.xlim(0, freq_show)

    plt.subplot(313)
    if method == 3:
        plt.plot(t, irfft(X), 'k--', label='recovered')
    else:
        plt.plot(t, ifft(X), 'k--', label='recovered')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print('Number of time points:', npnts, 'points')
    print('Number of points in frequency range:', len(freq), 'points')
    print('Frequency range:', min(freq), max(freq), 'Hz')
    return freq, powerspect
#%%
srate = sample_rate1
dt = 1/srate
duration = duration_of_sound1
t = np.arange(0,duration,dt)
npnts  = len(t)
x = samples1

freq, powerspect = fft_powerspectrum_plot(x, t, dt, npnts, 2000, 1, denoise=True)
#%%
# let us split the freq into 128 buckets

bins = np.linspace(0.0, 2000.0, 128 )
b0 = bins[0]
b1 = bins[1]
index = ([0]+[i for i, e in enumerate(freq) if e < b1])[-1]
# index = ([0]+[i for i, e in enumerate(listIWantToCheck[:indexOfLastThree]) if e < 2])[-1] + 1
p = sum(powerspect[:index])/len(powerspect[:index])

freq
