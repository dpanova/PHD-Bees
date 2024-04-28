#%%
from datasets import load_dataset, Audio
from ray import tune

minds = load_dataset("PolyAI/minds14", name="en-US", split="train")

minds = minds.train_test_split(test_size=0.2)
minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
labels = minds["train"].features["intent_class"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

#%%
from transformers import AutoFeatureExtractor

# model_id = "facebook/wav2vec2-base"
# model_id = "facebook/hubert-base-ls960"
model_id = 'MIT/ast-finetuned-audioset-10-10-0.4593'
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
#%%
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    return inputs

#%%
encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
encoded_minds = encoded_minds.rename_column("intent_class", "label")

#%%
import evaluate

accuracy = evaluate.load("accuracy")
#f1
import numpy as np


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)

#%%
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
)

#%%
training_args = TrainingArguments(
    output_dir='bla',
    do_train=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=0,
    learning_rate=tune.uniform(1e-5, 5e-5),
    per_device_train_batch_size=tune.choice([16, 32, 64]),
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    weight_decay = tune.uniform(0.0, 0.3),
    max_steps=-1,
    num_train_epochs=tune.choice([2, 3, 4, 5,10]),
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

#%%
def model_init():
    return AutoFeatureExtractor.from_pretrained(model_id,return_dict=True)
#%%

trainer = Trainer(
    # model=model,
    args=training_args,
    train_dataset=encoded_minds["train"],
    eval_dataset=encoded_minds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    model_init=model_init
)


trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=10 # number of trials
)
