from datasets import Audio
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, AutoFeatureExtractor
import wandb

from auxilary_functions import preprocess_function, compute_metrics
import os
#%%
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="wave2vec-animals -project"

# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
max_duration=5
model_id ='ardneebwar/wav2vec2-animal-sounds-finetuned-hubert-finetuned-animals'
batch_size = 8
gradient_accumulation_steps = 4
num_train_epochs = 10
warmup_ratio=0.1
logging_steps = 10
learning_rate = 3e-5
name='finetuned-wav2vec-bee-wandb'
#%%

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True)
# resample the data to have the same sampling rate as the pretrained model
sampling_rate = feature_extractor.sampling_rate
data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))


# create numeric labels
id2label_fn = data["train"].features[beeclass.bee_col].int2str

id2label_fn = data["train"].features[beeclass.bee_col].int2str
id2label = {
    str(i): id2label_fn(i)
    for i in range(len(data["train"].features[beeclass.bee_col].names))
}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)


# construct the model
model = AutoModelForAudioClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

# add the train arguments
model_name = model_id.split("/")[-1]

training_args = TrainingArguments(
    output_dir='models',
    report_to="wandb",
    # model_name=name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=warmup_ratio,
    logging_steps=logging_steps,
    fp16=True,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# encode the data
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
    compute_metrics=compute_metrics
)

trainer.train()

#%%
#train the model



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
