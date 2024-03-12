# The goal of this file is to split the wav files into bee or no bee sections
# library
import pandas as pd
import os
from pydub import AudioSegment
import time

#%%
# read in all necessary files
# read the annotation file
annotation_df = pd.read_csv('beeAnnotations_enhanced.csv')

# read all files in the directory and filter for wav extension
path = "data"
dir_list = os.listdir(path)
extension = ('.wav','.mp3')
extension_files_list = []
for file in dir_list:
    if file.endswith(extension):
        extension_files_list.append(file)
    else:
        continue

# Note: there were duplicated files which we investigated if they are duplicated and then removed them
# also, we realized that some files are mp3 not only wav
#%%
#TODO have just one folder - SplitData
# now we will create a loop for to read each file and split it into the bee and no-bee
bee_folder = 'data/bee/'
no_bee_folder = 'data/nobee/'

#Do we actually need this?

# loop over all files in the list of files
for f in extension_files_list:
    time.sleep(3)
    print(f)
    if f.__contains__('.wav'):
        wav_flag = True
        dir_file = f.replace('.wav', '')
    else:
        wav_flag = False
        dir_file = f.replace('.mp3', '')

    to_label_df = annotation_df[annotation_df['file name'] == dir_file]
    if len(to_label_df) == 0: # not working as expected
        print('no such file')


    # iterate over one recording. split accordingly and save the resulting files
    for inx, row in to_label_df.iterrows():
        print(inx)
        time.sleep(3)
        start_time = row['start'] * 1000 # note: package splits in milliseconds
        end_time = row['end'] * 1000 # note: package splits in milliseconds
        bee_label = row['label']
        file_index = row['index']
        if wav_flag:
            recording = AudioSegment.from_wav('data/' + f)
        else:
            recording = AudioSegment.from_mp3('data/' + f)
        new_recording = recording[start_time:end_time]
        new_recording_name = bee_label + '_index' + str(file_index) + '.wav'
        if bee_label == 'bee':
            new_recording.export(bee_folder + new_recording_name, format="wav")
        else:
            new_recording.export(no_bee_folder + new_recording_name, format="wav")

#%%
# This section is to investigate why not all files are split based on the annotation
bee_files = os.listdir(bee_folder)
nobee_files = os.listdir(no_bee_folder)

# check which files haven't been saved
# it looks like 350 are not saved for some reason
bee_files_index = [int(x.replace('bee_index','').replace('.wav','')) for x in bee_files]
bee_index_true = annotation_df.loc[annotation_df['label']=='bee','index']
len(list(set(bee_index_true).difference(set(bee_files_index)))) #194
bee_difference = list(set(bee_index_true).difference(set(bee_files_index)))
len(list(set(bee_files_index).difference(set(bee_index_true)))) #0

# let us see for the others
nobee_files_index = [int(x.replace('nobee_index','').replace('.wav','')) for x in nobee_files]
nobee_index_true = annotation_df.loc[annotation_df['label']=='nobee','index']
len(list(set(nobee_index_true).difference(set(nobee_files_index)))) #173
nobee_difference = list(set(nobee_index_true).difference(set(nobee_files_index)))
len(list(set(nobee_files_index).difference(set(nobee_index_true)))) #0


# check for the dir name possible issue

inspect_files = annotation_df.loc[annotation_df['index'].isin(bee_difference),'file name'].unique()
inspect_index = annotation_df.loc[annotation_df['file name'].isin(inspect_files),'index']
diff_index = bee_difference+nobee_difference
set(inspect_index).difference(set(diff_index))
set(diff_index).difference(set(inspect_index))

# ok, this looks to be the issue - the naming convention of a few of the files in the directory is different than those
# in the annotation file. The check if not working as expected since it is printing the issue in a bunch of files
# At this point we will remove those files from the analysis since it is not clear how to transform them to music files
# at a later stage we may contact the authors of the data https://filext.com/file-extension/LAB We did try connecting
# using Sonic vizualizer but it is not opening .lab files

for i in inspect_files:
    print(i)
    found_file = [x for x in dir_list if x.find(i) != -1]
    print(found_file)

# Ok, yes all files are indeed only lab files

