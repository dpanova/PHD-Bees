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

duration = bee.annotation_df.loc[bee.annotation_df['index']==1,'duration'][0]
dt = 1/sample_rate1
t = np.arange(0,duration,dt)
npnts  = len(t)
bee.binning(x=samples1, dt=dt, npnts = npnts)
# freq, powerspect = bee.freq_powerspect_func(x=samples1, dt=dt, npnts = npnts)
#%%
bins = np.linspace(0.0, 2000.0, 128 )
#bins_list = list(zip(bins[::2], bins[1::2]))
#bins_list = [((i), (i + 1) % len(bins)) for i in range(len(bins))]
bins_list = [(bins[i], bins[i + 1] ) for i in (range(len(bins)-1))]
pair = bins_list[0]
b0 = pair[0]
b1 = pair[1]
index = ([b0]+[i for i, e in enumerate(freq) if e < b1])[-1]
old_index = index
p = sum(powerspect[:index])/len(powerspect[:index])


#%%
transformed_x = list()
for pair in bins_list:
    b0 = pair[0]
    b1 = pair[1]
    index = ([b0]+[i for i, e in enumerate(freq) if e < b1])[-1]
    try:
        old_index
    except NameError:
        old_index = 0
    try:
        p = sum(powerspect[old_index:index])/len(powerspect[old_index:index])
    except:
        p=0
    transformed_x.append(p)
    old_index = index

