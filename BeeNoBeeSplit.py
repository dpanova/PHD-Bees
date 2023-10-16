# The goal of this file is to split the wav files into bee or no bee sections
# library
import pandas as pd
import os
from pydub import AudioSegment

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
# now we will create a loop for to read each file and split it into the bee and no-bee
bee_folder = 'data/bee/'
no_bee_folder = 'data/nobee/'

# loop over all files in the list of files
for f in extension_files_list:

    if f.__contains__('.wav'):
        wav_flag = True
        dir_file = f.replace('.wav', '')
    else:
        wav_flag = False
        dir_file = f.replace('.mp3', '')

    to_label_df = annotation_df[annotation_df['file name'] == dir_file]
    if len(to_label_df) == 0:
        print('no such file')
    # to_label_df.reset_index(inplace=True) , i don't think i need this since we need traceability

    # iterate over one recording. split accordingly and save the resulting files
    for inx, row in to_label_df.iterrows():
        start_time = row['start'] * 1000 # note: package splits in milliseconds
        end_time = row['end'] * 1000 # note: package splits in milliseconds
        bee_label = row['label']
        if wav_flag:
            recording = AudioSegment.from_wav('data/' + f)
        else:
            recording = AudioSegment.from_mp3('data/' + f)
        new_recording = recording[start_time:end_time]
        new_recording_name = bee_label + '_index' + str(inx) + '.wav'
        if bee_label == 'bee':
            new_recording.export(bee_folder + new_recording_name, format="wav")
        else:
            new_recording.export(no_bee_folder + new_recording_name, format="wav")



