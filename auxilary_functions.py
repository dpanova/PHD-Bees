import pandas as pd
import math
import os
import shutil
import evaluate
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score,make_scorer
import scipy as sp


def split_list(row, column_name):
    """
    The goal of this function is to split a column which consists of a list into several columns
    :param row: row from a data frame
    :type row: pd.Series
    :param column_name: name of the column withthe list
    :type column_name: str
    :return: pd.Series with the unpacked list
    :rtype: pd.Series
    """
    return pd.Series(row[column_name])


def file_name_extract(row, column_name1, column_name2):
    """
    The goal of this function is to extract the file name and add it to the DF
    :param row: row from a data frame
    :type row: pd.Series
    :param column_name1: column name which has a label
    :type column_name1: str
    :param column_name2: another column which has a label
    :type column_name2: str
    :return: extracted file name
    :rtype: str
    """
    if pd.isnull(row[column_name2]):
        label = row[column_name1]
    else:
        label = math.nan
    return label


def get_file_names(dir_name, specific_extension = False, extension=('.wav','.mp3')):
    """
    Create a list of files in a specific directory with specified extension
    :param dir_name: directory which contains the files
    :type dir_name: str
    :param specific_extension: specify if we would be looking for a specific extension files
    :param extension: tuple with strings indicating the file extension
    :type extension: tuple
    :return: a list of files
    :rtype: list
    """
    if type(dir_name) != str:
        raise ValueError(
            'Invalid dir_name type. It is type %s and expected type is str.' % type(dir_name).__name__)
    if type(specific_extension) != bool:
        raise ValueError(
            'Invalid specific_extension type. It is type %s and expected type is bool.' % type(specific_extension).__name__)
    if type(extension) != tuple:
        raise ValueError(
            'Invalid extension type. It is type %s and expected type is str.' % type(extension).__name__)
    for a in extension:
        if type(a) != str:
            raise ValueError(
                'Invalid string type. It is type %s and expected type is str.' % type(a).__name__)
    list_of_files = os.listdir(dir_name)
    extension_files_list = []
    if specific_extension:
        for file in list_of_files:
            if file.endswith(extension):
                extension_files_list.append(file)
            else:
                continue
    else:
        extension_files_list = list_of_files
    if len(extension_files_list)==0:
        raise ValueError('Extension list is empty. Please, check the %s folder' % dir)
    return extension_files_list

def clean_directory(path, folder=False):
    """
    Clean the directory
    :param path: the directory which needs to be cleaned
    :type folder: str
    :param folder: specifies if we need to remove folders
    :type folder: bool
    """
    if type(path) != str:
        raise ValueError(
            'Invalid path type. It is type %s and expected type is str.' % type(path).__name__)
    if type(folder) != bool:
        raise ValueError(
            'Invalid folder type. It is type %s and expected type is bool.' % type(folder).__name__)
    to_delete = get_file_names(path)
    for item in to_delete:
        if folder:
            shutil.rmtree(os.path.join(path+item))
        else:
            os.remove(item)

def compute_metrics(eval_pred, accuracy_metric ='accuracy'):
    """Computes accuracy on a batch of predictions"""
    #TODO add validations
    metric = evaluate.load(accuracy_metric)
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def preprocess_function(examples,feature_extractor, max_duration=10):
    #TODO add validations
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs

def citations(x):
    if isinstance(x, str):
        try:
            value = int([y for y in x.split(';') if y.find('Citation') != -1][0].split(' ')[0])
        except:
            value = 0
    else:
        value = 0
    return value

def recommendations(x):
    if isinstance(x, str):
        try:
            value = int([y for y in x.split(';') if y.find('Recommendations') != -1][0].split(' ')[0])
        except:
            value = 0
    else:
        value = 0
    return value

def reads(x):
    if isinstance(x, str):
        try:
            value = int([y for y in x.split(';') if y.find('Reads') != -1][0].split(' ')[0])
        except:
            value = 0
    else:
        value = 0
    return value

def cos_func(v1,v2):
    #TODO check if v1 and v2 are vectors?
    cosine_similarity = 1 - cosine(v1, v2)
    return(cosine_similarity)

def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.cosine):
    """
    Function to predict the DBSCAN Label
    :param dbscan_model: Initiated DBSCAN model
    :type dbscan_model: DBSCAN
    :param X_new: vectors on which to predict
    :type X_new: array-like
    :param metric: metric to calculate the similarity
    :type metric: function to calculate the similarity between two vectors
    :return: ist of the lables
    :rtype: array-like
    """
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

def my_custom_function(model, X):
    """
    Function to calculate the silhouette score for the DBSCAN clustering in order to optimize the DBSCAN algorithm
    :param model: DBSCAN model
    :type model: DBSCAN
    :param X: vectors on which to predict
    :type X: array-like
    :return: silhouette score
    :rtype: float
    """
    # for models that implement it, e.g. KMeans, could use `predict` instead
    #TODO update description and move to auxilary
    preds = dbscan_predict(model, X)
    return silhouette_score(X, preds) if len(set(preds)) > 1 else float('nan')

def include_tuple(element_list, row, t_limit=0.3):
    """
    Return the tuple if at least one of the elements is present
    :param element_list: list of elements to include
    :type element_list: list
    :param row: row of a data frame
    :type row: pd.Series
    :param t_limit: threshold to exclude elements if they don't have good cosine similarity with the include list
    :type t_limit: float
    :return: boolean indicating if the tuple should be included or not
    :rtype: bool
    """
    if type(element_list) != list:
        raise ValueError(
            'Invalid element_list type. It is type %s and expected type is list.' % type(element_list).__name__)
    if type(t_limit) != bool:
        raise ValueError(
            'Invalid t_limit type. It is type %s and expected type is bool.' % type(
                t_limit).__name__)

    exclude = False
    for element in element_list:
        if (element in row['key']) & (row['cos'] >= t_limit):
            exclude = True
    return exclude
