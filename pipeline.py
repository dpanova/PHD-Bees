from BeeData import BeeData
from BeeClassification import BeeClassification

slice= 5

#Get the data
beedata = BeeData()
beedata.annotation_data_creation()
beedata.data_quality(path='data',min_duration=float(slice))
beedata.time_slice(step = int(slice)*1000)
beedata.split_acoustic_data_sliced()
beedata.create_validate_data(sliced=True)

#%%
beeclass = BeeClassification()
# read and validate the annotation data
beeclass.read_annotation_csv()
#create the new label
beeclass.new_y_label_creation()

# split the data
beeclass.split_annotation_data()

#%%
#augment the data
beeclass.data_augmentation_df()

#%%
# RF results
acc_pca, precision_pca, recall_pca, misclassified_pca, pca_variance, forest_importances_pca = beeclass.random_forest_results()

acc, precision, recall, misclassified, forest_importances = beeclass.random_forest_results(do_pca=False)

#%%
#create data dictionary
data = beeclass.dataframe_to_datadict(beeclass.X_train_index,beeclass.X_test_index)


#%%
#train the model

trainer= beeclass.transformer_classification(data = data
                                             , max_duration=slice)
trainer.evaluate()



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
