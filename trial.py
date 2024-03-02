# import pandas as pd

from BeeClassification import Bee

bee = Bee()
# read and validate the annotation data
bee.read_annotation_csv()
#create the new label
bee.new_y_label_creation()
bee.acoustic_file_names()
# split the data
bee.split_annotation_data()

# transform data to dataset

df_tr = bee.dataframe_to_dataset(bee.X_train_index,split_type='train')

#TODO figure out how to store the data


#%%
# The goal here is to transform the annotated data set to the correct data type
import pandas as pd
from datasets import Audio, Dataset
train_dataset = pd.DataFrame({})

for train_index, row in bee.X_train_index.iterrows():
    dataset = pd.DataFrame({})
    print(row['index'])
    sample, sample_rate =  bee.file_read(row['index'])
    dataset['audio'] = [sample]
    dataset['sampling_rate'] = sample_rate
    dataset['train_index'] = train_index
    dataset['file_index'] = row['index']
    dataset['label'] = bee.y_train.loc[train_index, bee.y_col]
    train_dataset = pd.concat([train_dataset, dataset], axis = 0)

data = Dataset.from_pandas(train_dataset, split="train")

#%%
dataset['audio'] = [mypath+fol+audio_folder+f for f in os.listdir(mypath+fol+audio_folder) if os.path.isfile(os.path.join(mypath+fol+audio_folder, f))]
dataset['sentence'] = pd.read_csv(mypath+fol+'/transcriptions.txt', header=None)[0].apply(str)
dataset['path'] = dataset['audio']
train_dataset = pd.concat([train_dataset, dataset], axis=0)





#%%
# Resampling - most of the pretrained models are trained at 16000
y, sr = librosa.load(librosa.ex('trumpet'), sr=22050)

y_8k = librosa.resample(y, orig_sr=sr, target_sr=8000)

y.shape, y_8k.shape



#%%
func = 'mel spec'
aa = list()
for train_index, row in bee.X_train_index.iloc[:2,:].iterrows():
    a = bee.data_transformation_row((train_index, row, func))
    aa.append(a)




#%%

# augment the data
#bee.data_augmentation_df()

# need to normalize and shorten the augmented data


#transform the data
x_transformed = bee.data_transformation_df(bee.X_train_index,func = 'mel spec')
x_test_transformed = bee.data_transformation_df(bee.X_test_index,func = 'mel spec')
# train data
#acc, precision, recall, misclassified = bee.random_forest_results()

#%%
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers
import keras



 # transform y to categorical value
bee.y_train = bee.y_train.astype('category')
bee.y_train_code = bee.y_train.apply(lambda x: x.cat.codes)
y = keras.utils.to_categorical(bee.y_train_code, num_classes=3)

bee.y_test = bee.y_train.astype('category')
bee.y_test_code = bee.y_test.apply(lambda x: x.cat.codes)
y_test = keras.utils.to_categorical(bee.y_test_code, num_classes=3)

#%%
#maybe we don't need to power_to_db for the mel_spec
import numpy as np
#transform x
# features_convolution = x_transformed.iloc[:,2:]
# https://stackoverflow.com/questions/60157742/convolutional-neural-network-cnn-input-shape
# channel should be 3 for RGB
X_train= np.reshape(x_transformed.iloc[:,2:],(678,398, -1,1))
X_test= np.reshape(x_test_transformed.iloc[:,2:],(226,398, -1,1))

#%%
#create model
model = Sequential()#add model layers


model.add(Conv2D(398, (1, 1), input_shape=X_train.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(398,-1,1))) #here is the key to the shape
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(10, activation='softmax'))

# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
#
# #model.add(Dropout(0.2))
#
# #'''
# #'''
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# #'''
#
#
# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#
# #model.add(Dense(1000))#input_shape=features.shape[1:]
# model.add(Dense(64))#input_shape=features.shape[1:]
#
# model.add(Dense(10))
# model.add(Activation('softmax'))
# sgd = optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




model.fit(X_train, y, validation_data=(X_test, y_test), epochs=3)




#%%
for col in x_transformed.columns:
    if sum(x_transformed[col].isnull())>0:
        print(col)




#%%
import librosa
import numpy as np
# test the MFCC ones

# get the necessary indices to trace files easily
file_index = bee.X_train_index['index'][0]  # this index is necessary to ensure we have the correct file name (coming from the annotation file)
#label = y.loc[train_index, self.bee_col]
# check if the file from the annotation data exists in the folders

#identify the file name
file_name = [x for x in bee.accoustic_files if x.find('index' + str(file_index) + '.wav') != -1][0]
file_name = bee.acoustic_folder + file_name



# read the file
samples, sample_rate = librosa.load(file_name, sr=None, mono=True, offset=0.0, duration=None)

# check if the data extraction is correct through the duration of the sample
duration_of_sound = round(len(samples) / sample_rate, 2)

# why mean here? we need to experiment
mfcc = np.mean(librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=100).T, axis=0)
S = np.mean(librosa.feature.melspectrogram(y = samples, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128).T, axis =0)


#np.insert(mfcc,0,file_index)





#%%
# TODO
# 1. check which files had issues and why, we need to document that - it looks like we just haven't split this data at all, so we need to revisit how do we split data at all, maybe after comparing it with the directory ?
# 4. why don't we take the max in each bin?
# 5. check if the splitting of the data is correct - annotation df creation
# 6. we should check the fft part

#1901 is not correct, we need to investigate why the file is null

# check the duration for the misclassified ones, the hypothesis is that it is the short ones
# we can also check with different window functions and xgboost
# we should also time the algorithm
# resize to have the same length of samples
# should we normalize it?


# all files are not the same length
# normalization

#what is happenning with the swarming label?

#%%

# ADDITIONAL ANALYSIS

# validate that all files have the same sampling rate
#import pandas as pd
all_sample_rates = []
for train_index, row in bee.X_train_index.iloc[:2,:].iterrows():
    file_index = row['index']
    samples, sample_rate = bee.file_read(file_index)
    all_sample_rates.append(sample_rate)

for train_index, row in bee.X_test_index.iloc[:2,:].iterrows():
    file_index = row['index']
    samples, sample_rate = bee.file_read(file_index)
    all_sample_rates.append(sample_rate)

pd.DataFrame(all_sample_rates).value_counts()
# Conclusion -> there is just one record with sample rate which is 44100 -> resample
#  Nyquist limit -> half of the sampling rate -> more than 16kHz is the highest frequency this records can capture -> what is the highest expected bee Hz?
#%%
# validate  why swarming doesn't exist as a label
bee.annotation_df['action'].value_counts() # it exists here
bee.y_train['action'].value_counts()

bee.annotation_df.loc[bee.annotation_df['action']=='swarming','Dir Exist'].value_counts()
#Conclusion -> all swarming records are missing in the directory
