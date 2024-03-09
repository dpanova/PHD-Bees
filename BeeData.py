# import libraries
import logging
import math
import pandas as pd
import os
import numpy as np
from pydub import AudioSegment
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class BeeData:
    """"
    TODO update documentation
    :param annotation_file: name of the .mlf file to be read
    :type annotation_file: str
    :param start_col_name: name of the column with the start time
    :type start_col_name: str
    :param end_col_name: name of the column with the end time
    :type end_col_name: str
    :param label_col_name: name of the column with the bee/nobee label
    :type label_col_name: str
    :param file_col_name: name of the column with the file name
    :type file_col_name: str
    :param duration_col_name: column name for the duration
    :type duration_col_name: str
    """
    def __init__(self
                 ,logname='BeeData.log'
                 ,bee_col='label'
                 ,file_name = 'beeAnnotations_enhanced.csv'
                , annotation_file='beeAnnotations.mlf'
                , start_col_name='start'
                , end_col_name='end'
                , file_col_name='file name'
                ,duration_col_name = 'duration'
                 ,key_col_name='index'):
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)

        if not annotation_file.endswith('mlf'):
            raise ValueError(
                '%s input is not the correct type. It should be .mlf extension' % annotation_file)
        if type(annotation_file) != str:
            raise ValueError(
                'Invalid annotation_file type. It is type %s and expected type is str.' % type(annotation_file).__name__)
        if type(start_col_name) != str:
            raise ValueError(
                'Invalid start_col_name type. It is type %s and expected type is str.' % type(start_col_name).__name__)
        if type(end_col_name) != str:
            raise ValueError(
                'Invalid end_col_name type. It is type %s and expected type is str.' % type(end_col_name).__name__)
        if type(bee_col) != str:
            raise ValueError(
                'Invalid bee_col type. It is type %s and expected type is str.' % type(bee_col).__name__)
        if type(file_col_name) != str:
            raise ValueError(
                'Invalid file_col_name type. It is type %s and expected type is str.' % type(file_col_name).__name__)

        if type(duration_col_name) != str:
            raise ValueError(
                'Invalid duration_col_name type. It is type %s and expected type is str.' % type(duration_col_name).__name__)

        if type(key_col_name) != str:
            raise ValueError(
                'Invalid key_col_name type. It is type %s and expected type is str.' % type(key_col_name).__name__)


        self.annotation_df_data_quality = pd.DataFrame()
        self.annotation_df = pd.DataFrame()
        self.bee_col = bee_col
        self.file_name = file_name
        self.start_col_name = start_col_name
        self.annotation_file = annotation_file
        self.end_col_name = end_col_name
        self.file_col_name = file_col_name
        self.duration_col_name = duration_col_name
        self.key_col_name = key_col_name
    def split_list(self,row, column_name):
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

    def file_name_extract(self,row, column_name1, column_name2):
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

    def mlf_data_read(self):
        """
        Read a file with .mlf extension. The file is expected to have the following format:
        - acoustic file name to be on a new row
        - each column entry to be separated by a tab
        - each tow to end with a new row sign
        - columns - start, end (in min and seconds format) and label -bee and nobee
        :return: a pandas dataframe
        :rtype: pandas.DataFrame
        """


        # read the file
        with open(self.annotation_file) as f:
            lines = f.readlines()
        logging.info('File %s read.' % self.annotation_file)
        #transform to DF
        lines_df = pd.DataFrame(lines)
        logging.info('Data transformed from list of pandas DataFrame.')
        # the precessed file consists of one line. each column is seperated by tab, even at pd level
        # check if each line has tabs and finishes with new row sign
        perc_tabs = sum(lines_df.loc[:,0].str.contains('\t'))/len(lines_df)
        if perc_tabs == 0:
            raise ValueError("The provided files does not split columns with tabs deliminator as expected")
        if perc_tabs < 0.9:
            logging.info('We have less split observations by file.')

        #check if there is a new row sign to each entry, as expected
        perc_new_row = sum(lines_df.loc[:, 0].str.contains('\n')) / len(lines_df)
        if perc_new_row != 1.0:
            raise ValueError("The provided files does not end in each row with new row sign as expected.")

        # first we create a list of values for each row
        lines_df['all'] = lines_df[0].apply(lambda x: x.split('\t'))
        logging.info('Create a list by tabs.')
        # then we split the list into a separate columns
        lines_df = lines_df.apply(lambda x: self.split_list(x, 'all'), axis=1)
        logging.info('Split the list into different columns.')
        # check if the name of the file is on a new row, this should result in rows with NaN entries except the first column
        nan_df = pd.DataFrame(lines_df.isna().sum())
        if (nan_df.loc[0,0] != 0) or (nan_df.loc[1,0] == 0) or (nan_df.loc[2,0] == 0):
            raise ValueError("The provided files does not have the file name in a separate row.")

        # add column names
        lines_df.rename(columns={0: self.start_col_name, 1: self.end_col_name, 2: self.bee_col}, inplace=True)
        logging.info('Add column names.')
        # add the file name to each relevant row
        lines_df[self.file_col_name] = lines_df.apply(lambda x: self.file_name_extract(x, self.start_col_name, self.end_col_name), axis=1)
        lines_df[self.file_col_name].ffill(inplace=True)
        logging.info('Added file name as a separate column.')
        # remove empty rows, since the original file had rows containing only the file name
        lines_df = lines_df[(lines_df[self.file_col_name] != '.\n') & (~lines_df[self.end_col_name].isna())]

        # remove new line character
        lines_df[self.file_col_name] = lines_df[self.file_col_name].str.replace('\n', '')
        lines_df[self.bee_col] = lines_df[self.bee_col].str.replace('\n', '')

        #ensure that only bee and nobee are the labels
        if len(set(lines_df[self.bee_col].unique()).difference(set(['bee', 'nobee']))) != 0:
            raise ValueError("The provided files has different labels than bee and nobee.")

        # change data types
        lines_df[self.start_col_name] = lines_df[self.start_col_name].astype(float)
        lines_df[self.end_col_name] = lines_df[self.end_col_name].astype(float)

        # add the index since it will be easier to refer to the piece of recording later on
        lines_df.reset_index(inplace=True)
        logging.info('Data is read correctly.')
        return lines_df


    def calc_duration(self, df):
        """
        Calculates the duration between two time periods
        :param df: The dataframe to calculate duration in
        :type df: pandas.DataFrame
        :return: pd.Series containing the duration
        :rtype: pandas.Series
        """
        if type(df) != pd.core.frame.DataFrame:
            raise ValueError(
                'Invalid df type. df is type %s and expected type is pd.core.frame.DataFrame.' % type(df).__name__)

        duration = df[self.end_col_name] - df[self.start_col_name]
        logging.info('Duration is calculated.')
        return duration

    def create_actions(self, df, dict_actions):
        """
        Creates new columns with labels based on  the provided dictionary
        :param df: The dataframe to calculate duration in
        :type df: pandas.DataFrame
        :param dict_actions: dictionary with the new column names as the key and the string match to be searched.Note that each string should be in a list
        :type dict_actions: dict
        :return: data frame with the new columns
        :rtype: pandas.DataFrame
        """
        if type(df) != pd.core.frame.DataFrame:
            raise ValueError(
                'Invalid df type. df is type %s and expected type is pd.core.frame.DataFrame.' % type(df).__name__)

        if type(dict_actions) != dict:
            raise ValueError(
                'Invalid dict_actions type. It is type %s and expected type is str.' % type(dict_actions).__name__)
        for a in dict_actions.values():
            if type(a) != list:
                raise ValueError(
                    'Invalid string type. It is type %s and expected type is list.' % type(a).__name__)

        for label in dict_actions.keys():
            df[label] = df[self.file_col_name].str.lower().str.contains('|'.join(dict_actions[label]))
        logging.info('Action columns created.')
        return df



    def annotation_data_creation(self
                                 , special_action_column=True
                                 ,dict_actions = {'missing queen': ['missing queen', 'no_queen'],'active day': ['active - day'],'swarming': ['swarming']}
                                 ):
        """
        Function to create the dataset for annotation and save it locally under the specified file name.

        :param special_action_column: boolean which indicates if the special action column will be calculated
        :type special_action_column: bool
        :param dict_actions: dictionary with the new column names as the key and the string match to be searched.Note that each string should be in a list
        :type dict_actions: dict
        :return: pandas.DataFrame containing the filename, start time, end time, duration between both, bee/nobee label and action columns. The data is stores in the object.
        :rtype: pandas.DataFrame
        """

        if type(dict_actions) != dict:
            raise ValueError(
                'Invalid dict_actions type. It is type %s and expected type is str.' % type(dict_actions).__name__)
        if type(special_action_column) != bool:
            raise ValueError('special_action_column input is not the correct type. It is type %s, but it should be a boolean.' % type(special_action_column))

        # read the data
        data = self.mlf_data_read()
        #calculate the duration
        data[self.duration_col_name] = self.calc_duration(df=data)

        #add the action columns
        data = self.create_actions(df=data
                                  , dict_actions=dict_actions)
        #create a special action column
        if special_action_column:
            data['queen'] = (data[self.file_col_name].str.lower().str.contains('queenbee')) & (
                ~data[self.file_col_name].str.lower().str.contains('no'))
            logging.info('Created a special action column')

        self.annotation_df = data

        #save the data locally
        data.to_csv(self.file_name, index=False)

    def get_file_names(self, dir_name, extension=('.wav','.mp3')):
        """
        Create a list of files in a specific directory with specified extension
        :param dir_name: directory which contains the files
        :type dir_name: str
        :param extension: tuple with strings indicating the file extension
        :type extension: tuple
        :return: a list of files
        :rtype: list
        """
        if type(dir_name) != str:
            raise ValueError(
                'Invalid dir_name type. It is type %s and expected type is str.' % type(dir_name).__name__)
        if type(extension) != tuple:
            raise ValueError(
                'Invalid extension type. It is type %s and expected type is str.' % type(extension).__name__)
        for a in extension:
            if type(a) != str:
                raise ValueError(
                    'Invalid string type. It is type %s and expected type is str.' % type(a).__name__)
        list_of_files = os.listdir(dir_name)
        logging.info('Files in the directory are stores in a list - %s'% dir)
        extension_files_list = []
        for file in list_of_files:
            if file.endswith(extension):
                extension_files_list.append(file)
            else:
                continue
        if len(extension_files_list)==0:
            raise ValueError('Extension list is empty. Please, check the %s folder' % dir)
        return extension_files_list



    def data_quality(self
                     , path='data'
                     ,  min_duration=2.0):
        """
        :param path: directory path for the wav and mp3 files
        :type path: str
        :param min_duration: minimum duration of the file segment in order to make sense to work with it
        :type min_duration: float

        :return: data frame with data quality annotation_df_quality, stored in the object
        :rtype: pd.DataFrame
        """
        if type(path) != str:
            raise ValueError(
                'Invalid path type. It is type %s and expected type is str.' % type(path).__name__)
        if type(min_duration) != float:
            raise ValueError(
                'Invalid min_duration type. It is type %s and expected type is str.' % type(min_duration).__name__)



        #get the existing files in the directory
        self.extension_files_list = self.get_file_names(path)
        existing_files_df = pd.DataFrame()
        existing_files_df[self.file_col_name] = self.extension_files_list
        existing_files_df['Dir Exist'] = True
        existing_files_df[self.file_col_name] = existing_files_df[self.file_col_name].str.replace('.wav','').str.replace('.mp3','' )
        self.annotation_df = self.annotation_df.merge(existing_files_df, how='left', left_on=self.file_col_name,
                                                      right_on=self.file_col_name)
        self.annotation_df['Dir Exist'].fillna(False, inplace=True)
        self.annotation_df_data_quality = self.annotation_df[self.annotation_df['Dir Exist']]
        self.annotation_df_data_quality = self.annotation_df_data_quality[self.annotation_df_data_quality[self.duration_col_name] > min_duration]
        logging.info('Annotation data quality created.')

    def time_slice(self
                   ,nobee=False
                   ,step = 2000
                   ,start_sliced_col_name='start_sliced'
                   ,end_sliced_col_name='end_sliced'):
        """
        Function to split the annotation data into smaller segments based on the step size
        :param nobee: Boolean value to indicate if nobee files to be added in the data set. True means nobee files are included.
        :type nobee: bool
        :param step: time step in miliseconds
        :type step: float
        :param start_sliced_col_name: name of the column for the start sliced
        :type start_sliced_col_name: str
        :param end_sliced_col_name: name of the column for the end sliced
        :type end_sliced_col_name: str
        :return: pd.DataFrame with an annotation data which is sliced, stored in self.annotation_df_sliced
        :rtype: pd.DataFrame

        """
        if not nobee:
            annotation_df_sliced = self.annotation_df_data_quality[self.annotation_df_data_quality[self.bee_col] == 'bee']
        else:
            annotation_df_sliced = self.annotation_df_data_quality

        # loop over every row and split it appropriately
        new_df = pd.DataFrame()
        for idx, row in annotation_df_sliced.iterrows():
            start = row['start'] * 1000
            end = row['end'] * 1000
            time_range = np.arange(start, end, step).tolist()
            temp_df = pd.DataFrame()
            temp_df[start_sliced_col_name] = time_range[:-1]
            temp_df[end_sliced_col_name] = time_range[1:]
            temp_df[self.key_col_name] = row[self.key_col_name]
            new_df = pd.concat([new_df, temp_df])

        self.annotation_df_sliced = annotation_df_sliced.merge(new_df, on=self.key_col_name, how='outer')

        logging.info('Annotation data is sliced.')

    def split_acoustic_data_sliced(self):
        #TODO update here
        # loop over all files in the list of files
        files = self.annotation_df_sliced[self.file_col_name].unique()
        for f in files:
            time.sleep(3)
            print(f)
            if f.__contains__('.wav'):
                wav_flag = True
                dir_file = f.replace('.wav', '')
            else:
                wav_flag = False
                dir_file = f.replace('.mp3', '')

            to_label_df = self.annotation_df_sliced[self.annotation_df_sliced['file name'] == f]
            if len(to_label_df) == 0:  # not working as expected
                print('no such file')

            # iterate over one recording. split accordingly and save the resulting files
            for inx, row in to_label_df.iterrows():
                print(inx)
                time.sleep(3)
                start_time = row['start'] * 1000  # note: package splits in milliseconds
                end_time = row['end'] * 1000  # note: package splits in milliseconds
                bee_label = row['label']
                file_index = row['index']
                if wav_flag:
                    recording = AudioSegment.from_wav('data/' + f)
                else:
                    recording = AudioSegment.from_mp3('data/' + f)
                new_recording = recording[start_time:end_time]
                new_recording_name = bee_label + '_index' + str(file_index) + '.wav'
                if bee_label == 'bee':
                    new_recording.export(bee_folder + new_recording_name, format="wav")
                else:
                    new_recording.export(no_bee_folder + new_recording_name, format="wav")

