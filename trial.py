from BeeClassification import Bee

bee = Bee()
# read and validate the annotation data
bee.read_annotation_csv()
#create the new label
bee.new_y_label_creation()
# this should be first to integrate the changes
bee.acoustic_file_names() # here we need to figure out if we need it and why
# split the data
bee.split_annotation_data()


#%%

# This was working before and we are aiming at it
from BeeClassification import BeeNotBee
import time
bee = BeeNotBee()
# read and validate the annotation data
bee.read_annotation_csv()
# this should be first to integrate the changes
bee.acoustic_file_names()
bee.split_annotation_data()

acc, precision, recall, misclassified = bee.random_forest_results()



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

