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
from scipy.fft import fft, fftfreq
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
# transform the train set with fft and store it to a DF
for train_index,row in X_train_index.iterrows():
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
for train_index,row in X_train_index.iterrows():
    print(train_index)
    # get the necessary indices
    file_index = row['index']  # this index is necessary to ensure we have the correct file name
    print(file_index)
    label = y_train.loc[train_index, 'label']
    print(label)


# need to add column names and check if all rows have the same column count

