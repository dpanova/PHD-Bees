from BeeOrNotToBeeClassification import BeeNotBee

bee = BeeNotBee()
# read and validate the annotation data
bee.read_annotation_csv()

split_annotation_data


#%%
p = 1.35
if isinstance(p, float) and p>=0.0 and p<=1.0:
    print('yes')
else:
    print('no')


