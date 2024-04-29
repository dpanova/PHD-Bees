#TODO add the augmentation data as well and test it after that


from BeeData import BeeData

from BeeClassification import BeeClassification

#need to check the labels for this 60 seconds

slice= 5
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
beeclass.data_augmentation_df()

#WHY it doens't work for the augmented data? for Random Forest

#%%
data = beeclass.dataframe_to_datadict(beeclass.X_train_index,beeclass.X_test_index)


#%%

trainer= beeclass.transformer_classification(data = data
                                             , max_duration=slice)
trainer.evaluate()


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
param_dist = {'n_estimators': [20,30,40],
              'max_depth': [2,8,10,12,14,16,18,20]}

# param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}

rf = RandomForestClassifier()
rand_search = beeclass.best_model(model=rf, param_dist=param_dist)

# import numpy as np
# rand_search.fit(X_train[[x for x in X_train.columns if x not in ['train_index', 'file_index'] ]],
#                             np.array(beeclass.y_train).ravel())
#%%
# y_pred = rand_search.predict(X_test[[x for x in X_test.columns if x not in ['train_index', 'file_index'] ]])
#%%
#standardise features and then fit the to PCA
from sklearn.preprocessing import StandardScaler #TODO
x = X_train[[x for x in X_train.columns if x not in ['train_index', 'file_index'] ]]
y =  np.array(beeclass.y_train).ravel()
x = StandardScaler().fit_transform(x)

#%%
from sklearn.decomposition import PCA #TODO
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)
pca.explained_variance_ratio_ #array([44.13543137, 14.68972706]) #TODO
#%%
import pandas as pd
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
# finalDf = pd.concat([principalDf, y], axis = 1)
#%%
x_test = X_test[[x for x in X_test.columns if x not in ['train_index', 'file_index'] ]]
x_test = StandardScaler().fit_transform(x_test)
test_transformed = pca.transform(x_test)
principalDf_test = pd.DataFrame(data = test_transformed
             , columns = ['principal component 1', 'principal component 2'])
#%%
# rf = RandomForestClassifier()
# rand_search = beeclass.best_model(model=rf, param_dist=param_dist)

import numpy as np
rand_search.fit(principalDf,y)
y_pred = rand_search.predict(principalDf_test)
accuracy_score(beeclass.y_test, y_pred) #0.48283038501560877
precision_score(beeclass.y_test, y_pred, average='macro') #0.4424481013855393
recall_score(beeclass.y_test, y_pred, average='macro') #0.49636085296619653


#%%
y_test_df = pd.DataFrame(beeclass.y_test)
y_test_df['pred'] = y_pred
y_test_df['check'] = y_test_df[beeclass.y_col] == y_test_df['pred']
misclassified = y_test_df.loc[y_test_df[~y_test_df['check']].index, :].value_counts()

#%%
code_str = "sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r')"
cm = confusion_matrix(beeclass.y_test, y_pred)
beeclass.plot_figure(
    plot_title='cm_title'
    , file_title='cm_file_name.png'
    , plot_code=code_str
)

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
