from BeeData import BeeData

slice = 5
# Get the data
beedata = BeeData()
beedata.annotation_data_creation()
beedata.data_quality(path='data', min_duration=float(slice))
beedata.time_slice(step=int(slice) * 1000)
beedata.split_acoustic_data_sliced()
beedata.create_validate_data(sliced=True)

# %%
from BeeClassification import BeeClassification

beeclass = BeeClassification()
# read and validate the annotation data
beeclass.read_annotation_csv()
# create the new label
beeclass.new_y_label_creation()
# split the data
beeclass.split_annotation_data()
# augment the data
beeclass.data_augmentation_df()
# RF results with MFCC and PCA
acc_pca, precision_pca, recall_pca, misclassified_pca, pca_variance, forest_importances_pca = beeclass.random_forest_results()
# RF results with MFCC and NO PCA
acc, precision, recall, misclassified, forest_importances = beeclass.random_forest_results(do_pca=False)
# RF results with MEL SPEC and PCA
acc_pca_m, precision_pca_m, recall_pca_m, misclassified_pca_m, pca_variance_m, forest_importances_pca_m = beeclass.random_forest_results(
    func='mel spec')
# RF results with MEL SPEC and NO PCA
acc_m, precision_m, recall_m, misclassified_m, forest_importances_m = beeclass.random_forest_results(
    func='mel spec',
    do_pca=False)
# create data dictionary
data = beeclass.dataframe_to_datadict(beeclass.X_train_index, beeclass.X_test_index)
trainer = beeclass.transformer_classification(
    data=data,
    max_duration=slice)
trainer.evaluate()
