#Note: creating functions which will help in the creation of FFT composition of the bee files
import pandas as pd
import math
def split_list(row, column_name):
    #The goal of this function is to split a column which consists of a list into several columns
    return pd.Series(row[column_name])

def file_name_extract(row, column_name1, column_name2):
    #The goal of this function is to extract the file name and add it to the DF
    if pd.isnull(row[column_name2]):
        label = row[column_name1]
    else:
        label = math.nan
    return label
