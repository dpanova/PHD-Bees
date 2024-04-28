#TODO add the augmentation data as well and test it after that


from BeeData import BeeData

from BeeClassification import BeeClassification

#need to check the labels for this 60 seconds

slice= 30
# from BeeData import BeeData
#Get the data
beedata = BeeData()
beedata.annotation_data_creation()
beedata.data_quality(path='data',min_duration=float(slice))
beedata.time_slice(step = int(slice)*1000)
beedata.split_acoustic_data_sliced()
beedata.create_validate_data(sliced=True)

#%%


# from BeeClassification import BeeClassification
beeclass = BeeClassification()
# read and validate the annotation data
beeclass.read_annotation_csv()
#create the new label
beeclass.new_y_label_creation()

# split the data
beeclass.split_annotation_data()
#here should be added the data augmentation information
#%%
# beeclass.data_augmentation_df()

#WHY it doens't work for the augmented data? for Random Forest

#%%
data = beeclass.dataframe_to_datadict(beeclass.X_train_index,beeclass.X_test_index)


#%%

trainer= beeclass.transformer_classification(data = data
                                             , max_duration=slice)

#%%

#CHECK THE SPLIT, IT DOESN'T CREATE AN ACTUAL 30SEC FILE, WHICH MAYBE THE ISSUE

files = beedata.annotation_df_sliced[beedata.file_col_name].unique()
f = files[0]
to_label_df = beedata.annotation_df_sliced[beedata.annotation_df_sliced[beedata.file_col_name] == f]
from pydub import AudioSegment
recording = AudioSegment.from_wav('data/' + f + '.wav')
for inx, row in to_label_df.iterrows():
    start_time = row['start_sliced'] #here should be the sliced column
    end_time = row['end_sliced'] #here should be the sliced column
    file_index = row[beedata.key_col_name]
    new_recording = recording[start_time:end_time] #change this with a new package
    new_recording_name = beedata.acoustic_folder + 'index' + str(file_index) + '.wav'
    new_recording.export(new_recording_name, format="wav")
#%%



# data
max_duration = 30
model_id = 'facebook/hubert-base-ls960'
batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 1
warmup_ratio = 0.1
logging_steps = 10
learning_rate = 3e-5
name = 'finetuned-bee'
max_length = 1000
problem_type = "multi_label_classification"

#%%
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, HubertConfig
from datasets import Dataset, Audio, ClassLabel
#create the feature extractor
# config = HubertConfig.from_pretrained('facebook/hubert-base-ls960',
#                                       problem_type = 'multi_label_classification',
#                                       mask_time_length = 5,
#                                       mask_feature_length = 5,
#                                       is_encoder = True)

model_id = 'facebook/hubert-base-ls960'
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True, return_attention_mask=True, config=config)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True, return_attention_mask=True)



# resample the data to have the same sampling rate as the pretrained model
sampling_rate = feature_extractor.sampling_rate
data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))

#create numeric labels
id2label_fn = data["train"].features[beeclass.bee_col].int2str

id2label_fn = data["train"].features[beeclass.bee_col].int2str
id2label = {
    str(i): id2label_fn(i)
    for i in range(len(data["train"].features[beeclass.bee_col].names))
}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

#construct the model
model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

#add the train arguments
model_name = model_id.split("/")[-1]

training_args = TrainingArguments(
    "%s-%s" %(model_name,name),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=warmup_ratio,
    logging_steps=logging_steps,
    # fp16=True,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)
from auxilary_functions import get_file_names, clean_directory, compute_metrics, preprocess_function

#encode the data
data_encoded = data.map(
    preprocess_function,
    remove_columns=["audio", "file_index"],
    batched=True,
    batch_size=100,
    num_proc=1,
    fn_kwargs={"feature_extractor": feature_extractor, "max_duration": max_duration}
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=data_encoded["train"],
    eval_dataset=data_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()



#%%%
#QUESTION: WHY THE INPUT VALUES HAVE DIFFERENT SIZE? SHOULDN'T IT BE THE SAME?
x_list = []
for i in range(len(data_encoded['train'])):
    x = data_encoded['train'][i]['input_values']
    x_list.append(x)

#%%
#TEST IF THIS HAPPENS IN THE DATASET CREATION


data['train'][3]['audio']['array']
#Answer -> yes, it does
#BUT WHY?
#%%%
import pandas as pd
dataset = pd.DataFrame({})
split_folder = 'train/'
clean_directory(beeclass.datadict_folder + split_folder, folder=True)
#%%
import numpy as np
all_indices = np.array_split(beeclass.X_train_index.index, 10)
set_indices = all_indices[0]
#%%
dataset = pd.DataFrame({})
split_type = 'train'
for train_index, row in beeclass.X_train_index.loc[set_indices,].iterrows():

    temp_dataset = pd.DataFrame({})
    path, sample, sample_rate = beeclass.file_read(row['index'],output_file_name=True)

    temp_dataset['audio'] = [{'path':path,
                             'array':sample,
    'sampling_rate':sample_rate}]

    # to update here to have path as an array with everything below
    temp_dataset['train_index'] = train_index
    temp_dataset['file_index'] = row['index']
    if split_type == 'train':
        temp_dataset['label'] = beeclass.y_train.loc[train_index, beeclass.y_col]
    else:
        temp_dataset['label'] = beeclass.y_test.loc[train_index, beeclass.y_col]
    dataset = pd.concat([dataset, temp_dataset], axis=0)

data = Dataset.from_pandas(dataset, split=split_type)


#%%
#RANDOM FOREST
# transform and run RF
X_train = beeclass.data_transformation_df(beeclass.X_train_index,
                                          func = 'mfcc')
X_test = beeclass.data_transformation_df(beeclass.X_test_index,
                                          func = 'mfcc')
#%%
from random import randint
from sklearn.ensemble import RandomForestClassifier
param_dist = {'n_estimators': [2,8,10,20,30,40,50,60,70,80,90,100],
              'max_depth': [2,8,10,12,14,16,18,20]}
rf = RandomForestClassifier()
rand_search = beeclass.best_model(model=rf, param_dist=param_dist)

import numpy as np
rand_search.fit(X_train[[x for x in X_train.columns if x not in ['train_index', 'file_index'] ]],
                            np.array(beeclass.y_train).ravel())
#%%
y_pred = rand_search.predict(X_test[[x for x in X_test.columns if x not in ['train_index', 'file_index'] ]])


#%%
from sklearn.metrics import accuracy_score,precision_score,recall_score
acc = accuracy_score(beeclass.y_test, y_pred) #0.97
# calculate the precision score
precision = precision_score(beeclass.y_test, y_pred, average='macro')
recall = recall_score(beeclass.y_test, y_pred, average='macro')

#%%
#EXPERIMENT WITH DIFFERENT MODELS
# #%%
# model_list = [
#     "facebook/hubert-base-ls960"
#     ,'facebook/wav2vec2-base'
#     ,'MIT/ast-finetuned-audioset-10-10-0.4593'
#     ,'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
#     ,'ardneebwar/wav2vec2-animal-sounds-finetuned-hubert-finetuned-animals'
# ]
# #%%
# model_list = [
#     # "facebook/hubert-base-ls960"
#     # ,'facebook/wav2vec2-base'
#     # ,'MIT/ast-finetuned-audioset-10-10-0.4593'
#     # ,
#     'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
#     ,'ardneebwar/wav2vec2-animal-sounds-finetuned-hubert-finetuned-animals'
# ]
# # trainer_list=[]
# for m in model_list:
#     print(m)
#     trainer= beeclass.transformer_classification(data = data
#                                                  , max_duration=slice
#                                                  ,model_id = m)
#     trainer_list.append(trainer)
#     # trainer.evaluate()
