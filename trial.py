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
#transform the data
x_transformed = bee.data_transformation_df(bee.X_train_index,func = 'mel spec')
# train data
#acc, precision, recall, misclassified = bee.random_forest_results()
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
X_transformed = pool.map(bee.data_transformation_row, [(train_index, row, func) for train_index, row in bee.X_train_index.iterrows()])
# add the column names
cols = ['train_index', 'file_index']
max_length = max([len(x) for x in X_transformed if x is not None])
cols = cols + ['col' + str(x) for x in range(max_length - 2)]
# transform to data frame
X_df = pd.DataFrame(columns=cols)
for x in X_transformed:
    if len(x) != 2:
        X_df.loc[len(X_df)] = x
    else:
        x_updated = x + [None] * (max_length - len(x))
        X_df.loc[len(X_df)] = x_updated
logging.info('Whole data frame transformed.')




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

