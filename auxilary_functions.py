import pandas as pd
import math
import os
import shutil
import evaluate
import numpy as np
from scipy.spatial.distance import cosine
from io import BytesIO
from matplotlib import pyplot as plt



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

def pd_to_tuple(df,col):
    """
    Converts pd.dataframe.value_counts() to tuple for pdf table ingestion.
    :param df: pandas dataframe
    :type df: pd.DataFrame
    :param col: column name for the value_counts
    :type col: str
    :return: tuple of tuples
    :rtype: tuple
    """

    if type(col) != str:
        raise ValueError(
            'Invalid string type. It is type %s and expected type is str.' % type(col).__name__)
    if type(df) != pd.DataFrame:
        raise ValueError(
            'Invalid input type. It is type %s and expected type is pandas,DataFrame.' % type(df).__name__)
    pandas_table = pd.DataFrame(df[col].value_counts())

    pandas_table.reset_index(inplace=True)
    pandas_table[col] = pandas_table[col].astype('str')
    pandas_table['count'] = pandas_table['count'].astype('str')
    cols = tuple(pandas_table.columns)
    table_data = [tuple(x) for x in pandas_table.to_numpy()]
    table_data.insert(0,cols)
    table_data = tuple(table_data)
    return table_data

#TODO update the function descriptions and validation
def normal_text(text, pdf, x=5, italics=False):
    """
    Generates pdf normal multi-wor text
    :param text: the text
    :type  text: str
    :param pdf: FPDF instance
    :type  pdf: FPDF
    :param x: space after the text
    :type  x: int
    :param italics: boolean to indicate if text should be italics
    :return: pdf generated text
    """
    if italics:
        pdf.set_font('Arial', size=10, style='I')
    else:
        pdf.set_font('Arial', size=10)
    pdf.multi_cell(w=180, h=5, txt=text)
    pdf.ln(x)


def start_page(pdf):
    pdf.add_page()
    pdf.set_margins(10, 10, 10)


def h0(text, pdf, x=20):
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(w=180, h=10, txt=text, align='C')
    pdf.ln(x)


def h1(text, pdf, x=10):
    pdf.set_font('Arial', size=16, style='B')
    pdf.cell(w=40, h=10, txt=text)
    pdf.ln(x)


def h2(text, pdf, x=10):
    pdf.set_font('Arial', size=12, style='B')
    pdf.cell(w=40, h=10, txt=text)
    pdf.ln(x)


def pdf_table(table_data, pdf, x=10, width=40, cols=(20, 20)):
    with pdf.table(width=width, col_widths=cols, text_align="CENTER") as table:
        for data_row in table_data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.ln(x)


def pdf_graph(pdf, x, y, w, h, with_code=True, plot_code='', filename=''):
    if with_code:
        plt.figure()
        exec(plot_code)
        # Converting Figure to an image:
        img_buf = BytesIO()  # Create image object
        plt.savefig(img_buf, dpi=100)  # Save the image
        pdf.image(img_buf, x=x, y=y, w=w, h=h)
    else:
        pdf.image(filename, x=x, y=y, w=w, h=h)

