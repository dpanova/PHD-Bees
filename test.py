import pandas as pd

from BeeData import BeeData
bee = BeeData()
bee.annotation_data_creation()
bee.data_quality('data',2.0)
bee.time_slice()
bee.split_acoustic_data_sliced()

#%%





#%%
files_list = bee.get_file_names(bee.acoustic_folder)
for item in files_list:
    os.remove(item)
    # shutil.rmtree(os.path.join(bee.acoustic_folder+item))

#%%
bee.annotation_df[['missing queen', 'queen', 'active day', 'swarming', 'file name']].groupby(
    ['missing queen', 'queen', 'active day', 'swarming'], as_index=False).nunique()
# data is balanced for missing queen and queen, we have 15 active days files and only 2 swa

#%%



#%%



if wav_flag:
    recording = AudioSegment.from_wav('data/' + f)
else:
    recording = AudioSegment.from_mp3('data/' + f)

for a in list(time_range):
    new_recording = recording[start_time:end_time]
