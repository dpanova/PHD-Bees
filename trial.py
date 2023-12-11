from BeeOrNotToBeeClassification import BeeNotBee

bee = BeeNotBee()
# read and validate the annotation data
bee.read_annotation_csv()

bee.split_annotation_data()



#%%
import librosa
import numpy as np
file_name1 = 'data/bee/bee_index1.wav'
samples1, sample_rate1 = librosa.load(file_name1, sr=None, mono=True, offset = 0.0, duration=None)
#a = bee.harley_transformation_with_window(samples1)
#%%
duration = bee.annotation_df.loc[bee.annotation_df['index']==1,'duration'][0]
dt = 1/sample_rate1
t = np.arange(0,duration,dt)
npnts  = len(t)
#bee.freq_powerspect_func(x=samples1,dt=dt, npnts=npnts)
a = bee.binning(x=samples1, dt=dt, npnts = npnts,n_bins='a')

#%%
bee_files = os.listdir(os.getcwd())
