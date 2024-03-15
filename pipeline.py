from BeeData import BeeData
from BeeClassification import BeeClassification

#%%
from BeeData import BeeData
#Get the data
beedata = BeeData()
beedata.annotation_data_creation()
beedata.data_quality('data',30.0) #This could be the investigation part - how long should the recording be
beedata.time_slice(step = 30000)
beedata.split_acoustic_data_sliced()
beedata.create_validate_data() # we need to save it locally at the end


#%%

from BeeClassification import BeeClassification
beeclass = BeeClassification()
# read and validate the annotation data
beeclass.read_annotation_csv()
#create the new label
beeclass.new_y_label_creation()
# split the data
beeclass.split_annotation_data()

# data = beeclass.dataframe_to_dataset(beeclass.X_train_index,beeclass.X_test_index)
beeclass.dataframe_to_datadict(beeclass.X_train_index,beeclass.X_test_index)
# beeclass.datadict_creation()

#%%
beeclass.transformer_classification(data = beeclass.datadict_data)


#%%

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import Dataset, Audio
from auxilary_functions import get_file_names, clean_directory, compute_metrics, preprocess_function
# model.config.to_json_file("config.json")

#%%
# create the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    'facebook/hubert-base-ls960', do_normalize=True, return_attention_mask=True
)
# resample the data to have the same sampling rate as the pretrained model
sampling_rate = feature_extractor.sampling_rate
data = beeclass.datadict_data.cast_column("audio",
                        Audio(sampling_rate=sampling_rate))  # TODO maybe it is good to save the data within the object

# create numeric labels
id2label_fn = beeclass.datadict_data["train"].features[beeclass.bee_col].int2str

id2label = {
    str(i): id2label_fn(i)
    for i in range(len(beeclass.datadict_data["train"].features[beeclass.bee_col].names))
}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)

# construct the model
model = AutoModelForAudioClassification.from_pretrained(
    'facebook/hubert-base-ls960',
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)

model_name = 'facebook/hubert-base-ls960'.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
# num_train_epochs = 10
num_train_epochs = 7 #playaround with the learning rate for optimization

#playaround with the warmup ratio
#TODO - for optimization, playaround with the learning rate, warmup ration, num epochs and accuracy
#TODO - test for the min length of the recording
#TODO - add the augmented data
#TODO - understand what is loss and grad_norm

training_args = TrainingArguments(
    f"{model_name}-finetuned-bee",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    # learning_rate=5e-5,
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

#encode the data
# TODO this may not work
data_encoded = beeclass.datadict_data.map(
    preprocess_function,
    remove_columns=["audio", "file_index"],
    batched=True,
    batch_size=100,
    num_proc=1,
    fn_kwargs={"feature_extractor": feature_extractor, "max_duration": 30}
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=data_encoded["train"],
    eval_dataset=data_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics

)

trainer.train()

#with these parametersm we have shano results but it runs properly



