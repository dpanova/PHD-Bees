#TODO add the augmentation data as well and test it after that

from BeeData import BeeData

from BeeClassification import BeeClassification

#need to check the labels for this 60 seconds

# from BeeData import BeeData
#Get the data
beedata = BeeData()
beedata.annotation_data_creation()
beedata.data_quality(path='data',min_duration=float(30))
beedata.time_slice(step = int(30)*1000)
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
beeclass.data_augmentation_df(N=1) #does not clean correctly the folder, to check
# we should play around with N to remove duplictaive index

# pd.DataFrame(beeclass.X_train_index.index).value_counts()

#%%
data = beeclass.dataframe_to_datadict(beeclass.X_train_index,beeclass.X_test_index)

#%%
beeclass.transformer_classification(data = data)

#%%
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from datasets import Dataset, Audio, ClassLabel
model_id ='facebook/hubert-base-ls960'

#create the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    return_attention_mask=True
)
# resample the data to have the same sampling rate as the pretrained model
sampling_rate = feature_extractor.sampling_rate
data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))

#create numeric labels
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
    id2label=id2label
)

#add the train arguments
#TODO we need to add them as an input to the function
model_name = model_id.split("/")[-1]
batch_size = 32
gradient_accumulation_steps = 4
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-bee",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=10,
    # torch_compile=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    # fp16=True,
    push_to_hub=False,
    # dataloader_pin_memory=False
)
from auxilary_functions import get_file_names, clean_directory, compute_metrics, preprocess_function
#encode the data
max_duration = 30
# TODO this may not work
data_encoded = data.map(
    preprocess_function,
    remove_columns=["audio", "file_index"],
    batched=True,
    batch_size=100,
    num_proc=1,
    fn_kwargs={"feature_extractor": feature_extractor, "max_duration": max_duration}
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_encoded["train"],
    eval_dataset=data_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()
