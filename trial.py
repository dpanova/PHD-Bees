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
#%%
# transform data to dataset
train, test = bee.dataframe_to_datadict(bee.X_train_index,bee.X_test_index)



#%%
import datasets
#add this to the common function so that we end up with one thing
data = datasets.DatasetDict(
    {
        "train": train,
        "test": test,
    }
)

#%%
# check how the data looks like
train[0]
#%%
#are our features human readable?


# model_id = 'ardneebwar/wav2vec2-animal-sounds-finetuned-hubert-finetuned-animals'
#note the first model is fine-tuned from this
model_id = 'facebook/hubert-base-ls960'

# load predefined model
from transformers import AutoFeatureExtractor


feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
#%%

#resample to have the same sampling rate
sampling_rate = feature_extractor.sampling_rate
from datasets import Audio

#this is the issue but why
data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))

data['train'][0]['audio']
#%%
import numpy as np


sample = data["train"][0]["audio"]
print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")
inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])

print(f"inputs keys: {list(inputs.keys())}")

print(
    f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}"
)

#good varience

#%%
#fientune

id2label_fn = data["train"].features["label"].int2str


id2label = {
    str(i): id2label_fn(i)
    for i in range(len(data["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}


num_labels = len(id2label)


#%%
from transformers import AutoModelForAudioClassification
model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

#%%
from transformers import TrainingArguments

model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-bee",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    push_to_hub=False,
    dataloader_pin_memory=False
)
#%%

import evaluate
import numpy as np

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)
#%%
max_duration = 2.0


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs

#%%

data_encoded = data.map(
    preprocess_function,
    remove_columns=["audio", "file_index"],
    batched=True,
    batch_size=100,
    num_proc=1,
)
data_encoded

#%%
from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=data_encoded["train"],
    eval_dataset=data_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()




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
