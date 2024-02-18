import pandas as pd

from BeeClassification import Bee

bee = Bee()
# read and validate the annotation data
bee.read_annotation_csv()
#create the new label
bee.new_y_label_creation()
# this should be first to integrate the changes
bee.acoustic_file_names() # here we need to figure out if we need it and why
# split the data
bee.split_annotation_data()
# augment the data




#transform the data
#x_transformed = bee.data_transformation_df(bee.X_train_index,func = 'mel spec')
# train data
#acc, precision, recall, misclassified = bee.random_forest_results()
#%%
#TODO absract this piece of code because it is in the

# get the necessary indices to trace files easily
file_index = bee.X_train_index.iloc[0,0]
train_index = bee.X_train_index.index[0]
# this index is necessary to ensure we have the correct file name (coming from the annotation file)
label = bee.y_train.loc[train_index, bee.y_col]
# check if the file from the annotation data exists in the folders

# identify the file name
file_name = [x for x in bee.accoustic_files if x.find('index' + str(file_index) + '.wav') != -1][0]
file_name = bee.acoustic_folder + file_name

# read the file
samples, sample_rate = librosa.load(file_name, sr=None, mono=True, offset=0.0, duration=None)

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])
import soundfile as sf
# Augment/transform/perturb the audio data
augmented_samples = augment(samples=samples, sample_rate=sample_rate)
augment_folder = 'data/augment/'

bee.X_augmented_train = pd.DataFrame(columns=[bee.x_col])
bee.y_augmented_train = pd.DataFrame(columns=[bee.y_col])
# Do we need this at all?
# recording = AudioSegment.from_wav('data/' + f)
augmented_file_index = file_index+10000
augmented_file_name = 'index' + str(augmented_file_index) + '.wav'
augmented_train_index = train_index + 10000
sf.write(augment_folder+'aug1.wav',augmented_samples,sample_rate)
bee.X_augmented_train.loc[augmented_train_index,bee.x_col] = augmented_file_index
bee.y_augmented_train.loc[augmented_train_index,bee.y_col] = label



# TODO
# not most optimal code
# clear the folder if there are other files
# save wav files into a separate folder
# save the index to another X train and the associated labels into the other y_train
# number of aumentations ? stratified?





#%%
# import multiprocessing as mp
# pool = mp.Pool(processes=mp.cpu_count())
# # TODO once, we add a new function, we need to check if this is working or not
func = 'mfcc'
aa= list()
for train_index, row in bee.X_train_index.iloc[:2,:].iterrows():
    print(train_index)
    a= bee.data_transformation_row((train_index, row, func))
    aa.append(a)

#%%





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

