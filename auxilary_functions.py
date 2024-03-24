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

def compute_metrics(eval_pred):
    """
    Computes accuracy on a batch of predictions
    :param eval_pred: predictions returned by the HuggingFace model
    :type eval_pred: array like
    :return: the computed evaluation metric
    :rtype: float
    """

    if not pd.api.types.is_list_like(eval_pred):
        raise ValueError(
            'Invalid eval_pred type. It is type %s and expected type is array-like.' % type(eval_pred).__name__)

    metric = evaluate.load('accuracy')
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def preprocess_function(examples,feature_extractor, max_duration=10):
    """
    Function to preprocess the audio data to a predefined sampling rate and duration
    :param examples: datadict examples with 'audio' key
    :type examples: datadict
    :param feature_extractor: AutoFeatureExtractor from pretrained model
    :type feature_extractor: AutoFeatureExtractor
    :param max_duration: maximum seconds recording
    :type max_duration: int
    :return: the preprocessed data as AutoFeatureExtractor
    :rtype: AutoFeatureExtractor
    """
    if type(max_duration) != int:
        raise ValueError(
            'Invalid max_duration type. It is type %s and expected type is int.' % type(max_duration).__name__)


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
    """
    Function to extract the number of citations from a string
    :param x: string which has a key word - Citations to extarct the info from
    :type x: str
    :return: Number of citations
    :rtype: int
    """
    if isinstance(x, str):
        try:
            value = int([y for y in x.split(';') if y.find('Citation') != -1][0].split(' ')[0])
        except:
            value = 0
    else:
        value = 0
    return value

def recommendations(x):
    """
    Function to extract the number of Recommendations from a string
    :param x: string which has a key word - Recommendations to extarct the info from
    :type x: str
    :return: Number of Recommendations
    :rtype: int
    """
    if isinstance(x, str):
        try:
            value = int([y for y in x.split(';') if y.find('Recommendations') != -1][0].split(' ')[0])
        except:
            value = 0
    else:
        value = 0
    return value

def reads(x):
    """
    Function to extract the number of Reads from a string
    :param x: string which has a key word - Reads to extarct the info from
    :type x: str
    :return: Number of Reads
    :rtype: int
    """
    if isinstance(x, str):
        try:
            value = int([y for y in x.split(';') if y.find('Reads') != -1][0].split(' ')[0])
        except:
            value = 0
    else:
        value = 0
    return value

def cos_func(v1,v2):
    """
    Function to calculate the cosine similiarity between two array-like variables
    :param v1: array-like variable
    :type v1: array-like
    :param v2: array-like variable
    :type v2: array-like
    :return: cosine similarity between teh two
    :rtype: float
    """
    if not pd.api.types.is_list_like(v1):
        raise ValueError(
            'Invalid v1 type. It is type %s and expected type is array-like.' % type(v1).__name__)
    if not pd.api.types.is_list_like(v2):
        raise ValueError(
            'Invalid v2 type. It is type %s and expected type is array-like.' % type(v2).__name__)


    cosine_similarity = 1 - cosine(v1, v2)
    return cosine_similarity

def cos_sim_func(pair,embedding_list):
    """
    Function to calculate the similarity between two embeddings based on cosine similarity
    :param pair: tuple with the index of the two embeddings
    :type pair: tuple
    :param embedding_list: list of the embedding vectors
    :type embedding_list: array-like
    :return: dictionary with pair0, pair1 and cos
    :rtype: dict
    """
    if type(pair) != tuple:
        raise ValueError(
            'Invalid pair type. It is type %s and expected type is tuple.' % type(pair).__name__)
    if not pd.api.types.is_list_like(embedding_list):
        raise ValueError(
            'Invalid embedding_list type. It is type %s and expected type is array-like.' % type(embedding_list).__name__)

    a0 = np.array(embedding_list[pair[0]])
    a1 = np.array(embedding_list[pair[1]])
    cos_sim = cos_func(a0, a1)
    temp_dict = {'pair0': pair[0], 'pair1': pair[1], 'cos': cos_sim}
    return temp_dict

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
    if not pd.api.types.is_list_like(X):
        raise ValueError(
            'Invalid X type. It is type %s and expected type is array-like.' % type(X).__name__)
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
    if type(t_limit) != float:
        raise ValueError(
            'Invalid t_limit type. It is type %s and expected type is bool.' % type(
                t_limit).__name__)

    exclude = False
    for element in element_list:
        if (element in row['key']) & (row['cos'] >= t_limit):
            exclude = True
    return exclude
