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

# transform the train set with fft and store it to a DF
train_index = X_train_index.index[0]
file_index = X_train_index.loc[train_index,'index']
label = y_train.loc[train_index,'label']

bee_folder = 'data/bee/'
no_bee_folder = 'data/nobee/'

bee_files = os.listdir(bee_folder)
nobee_files = os.listdir(no_bee_folder)

if label =='bee':


else:
    folder = 'data/nobee/'


a = [file for file in bee_files if str(file_index) in file]
#2239

# here we see an issue, not all data points are saved
