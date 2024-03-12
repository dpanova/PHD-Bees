import pandas as pd
import math
import os
import shutil


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
