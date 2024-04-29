# libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from auxilary_functions import get_file_names, clean_directory, compute_metrics, preprocess_function
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import logging
import numpy as np
from scipy.fft import fft,rfftfreq
import librosa
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TanhDistortion, GainTransition,AirAbsorption
import soundfile as sf
from datasets import Dataset, Audio
import datasets
import time
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class BeeClassification:
    """BeeNotBee class ....
    Class to conduct modelling based on labeled audio data, using HuggingFace transformers and Random Forest
    :param annotation_path: path to the annotation data which holds the labeled audio data
    :type annotation_path: str
    :param y_col: column label for the dependent variable in the annotation file, the label
    :type y_col: str
    :param x_col: column label for the independent variable in the annotation file. It is used to read the wav file.
    :type x_col: str
    :param acoustic_folder: folder name where acoustic files are stored
    :type acoustic_folder: str
    :bee_col: column which indicates the dependent variable in the annotation. In the beginning bothe bee_col and y_col are the same, but y_col can be updated during specific actions.
    :type bee_col: str
    :param acoustic_folder: folder name where the split files are stored
    :type acoustic_folder: str
    :param augment_folder: folder name where the augment files are stored
    :type augment_folder: str
    :param datadict_folder: folder name where the datadict files are stored
    :param logname: path to the log file
    :type logname: str
    :return: BeeClassification object
    :rtype: BeeClassification
    """

    def __init__(self
                 ,annotation_path ='beeAnnotations_enhanced.csv'
                 ,annotation_dtypes_path = 'annotation_data_types.csv'
                 ,x_col = 'index'
                 ,y_col = 'label'
                 ,bee_col = 'label'
                 ,logname='bee.log'
                 ,acoustic_folder = 'data/SplitData/'
                 ,augment_folder = 'data/augment/'
                 ,datadict_folder = 'data/DataDict/'):
        self.X_test = None
        self.X_train = None
        self.forest_importances = None
        self.annotation_path =annotation_path
        self.annotation_df = pd.DataFrame()
        self.annotation_dtypes_path = annotation_dtypes_path
        self.x_col = x_col
        self.y_col = y_col
        self.bee_col = bee_col
        self.acoustic_folder = acoustic_folder
        self.accoustic_files = get_file_names(self.acoustic_folder)
        self.augment_folder = augment_folder
        self.augmented_df = pd.DataFrame()
        self.datadict_folder = datadict_folder
        self.datadict_data = None
        self.X_train_index = None
        self.X_test_index = None
        self.y_train = None
        self.y_test = None
        self.augmented_files = None
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)

        #validate the inputs
        if type(annotation_path) != str:
            raise ValueError(
                'Invalid annotation_path type. It is type %s and expected type is str.' % type(annotation_path).__name__)
        if not annotation_path.endswith('csv'):
            raise ValueError(
                '%s input is not the correct type. It should be .csv extension' % annotation_path)
        if type(annotation_dtypes_path) != str:
            raise ValueError(
                'Invalid annotation_dtypes_path type. It is type %s and expected type is str.' % type(
                    annotation_dtypes_path).__name__)
        if not annotation_dtypes_path.endswith('csv'):
            raise ValueError(
                '%s input is not the correct type. It should be .csv extension' % annotation_dtypes_path)
        if type(y_col) != str:
            raise ValueError(
                'Invalid y_col type. It is type %s and expected type is str.' % type(
                    y_col).__name__)
        if type(bee_col) != str:
            raise ValueError(
                'Invalid bee_col type. It is type %s and expected type is str.' % type(
                    bee_col).__name__)
        if type(logname) != str:
            raise ValueError(
                'Invalid logname type. It is type %s and expected type is str.' % type(
                    logname).__name__)
        if type(acoustic_folder) != str:
            raise ValueError(
                'Invalid acoustic_folder type. It is type %s and expected type is str.' % type(
                    acoustic_folder).__name__)
        if type(augment_folder) != str:
            raise ValueError(
                'Invalid augment_folder type. It is type %s and expected type is str.' % type(
                    augment_folder).__name__)
        if type(datadict_folder) != str:
            raise ValueError(
                'Invalid datadict_folder type. It is type %s and expected type is str.' % type(
                    datadict_folder).__name__)

    def read_annotation_csv(self):
        """
        Read annotation data
        :return: pandas data frame with the annotations
        """

        try:
            self.annotation_df = pd.read_csv(self.annotation_path)
            logging.info('Annotation data is read successfully')
            self.validate_annotation_csv()
        except:
            logging.warning('Annotation data is NOT read successfully')

    def new_y_label_creation(self, old_col=['missing queen', 'queen',
       'active day', 'swarming'], new_col='action'):
        """
        Transform boolean columns into one column.
        :param old_col: list of column names
        :type old_col: list
        :param new_col: new column name
        :type new_col: str
        :return: updated annotation dataframe with a new column and updated y_col in the class
        """

        if type(old_col) != list:
            raise ValueError('old_col input is not the correct type. It is type %s, but it should be a list.' %type(old_col))
        if type(new_col) != str:
            raise ValueError('new_col input is not the correct type. It is type %s, but it should be a str.' %type(new_col))
        if not set(old_col).issubset(self.annotation_df.columns):
            raise ValueError('%s are not part of the annotation_df columns' %old_col)
        else:
            self.annotation_df[new_col] = [row.idxmax() for idx,row in self.annotation_df[old_col].iterrows()]
            self.y_col = new_col
            logging.info('New label created and assigned to the class')


    def validate_annotation_csv(self):
        """"
        Validate annotation data
        :return: warning if the data is as expected
        """
        dtypes_df = pd.read_csv(self.annotation_dtypes_path)
        cols = dtypes_df['col_name']
        if len(set(cols).difference(set(self.annotation_df))) == 0:
            logging.info("All columns are present correctly")
        else:
            logging.warning("NOT all columns are present correctly")

        # check that the data types are as expected
        data_type = pd.DataFrame(self.annotation_df.dtypes)
        data_type.reset_index(inplace=True)
        data_type.columns = dtypes_df.columns
        check_df = data_type.merge(dtypes_df,
                                              right_on=dtypes_df.columns[0],
                                              left_on=dtypes_df.columns[0],
                                              how='left')
        check_df['type check'] = check_df[dtypes_df.columns[1] + '_x'] == check_df[
            dtypes_df.columns[1] + '_y']
        if sum(check_df['type check']) == len(check_df):
            logging.info("All data is in the correct type")
        else:
            logging.error("NOT all data is in the correct type")




    def split_annotation_data(self,perc=0.25, stratified = True):
        """
        Split the annotation data into train and test based on the y_col and x_col values. Save the csv files.
        :param perc: test split percentage, a number between 0.0 and 1.0
        :type perc: float
        :param stratified: Boolean value to indicate if the split should be stratified
        :type stratified: bool
        :return:annotation_df_updated with the final data which is split. X train, X test, y train and y test pandas data frames
        """
        if type(perc) != float:
            raise ValueError(
                'perc input is not the correct type. It is type %s, but it should be a float.' % type(perc))
        if perc < 0.0:
            raise ValueError('perc should be bigger than 0.')
        if perc > 1.0:
            raise ValueError('perc should be smaller than 1.')
        if type(stratified) != bool:
            raise ValueError(
                'stratified input is not the correct type. It is type %s, but it should be a boolean.' % type(
                    stratified))
        if self.x_col not in self.annotation_df:
            raise ValueError('x_col not in the annotation_df. Change x_col.')
        if self.y_col not in self.annotation_df:
            raise ValueError('y_col not in the annotation_df. Change y_col.')

        if stratified:
            self.X_train_index, self.X_test_index, self.y_train, self.y_test = train_test_split(
                self.annotation_df[[self.x_col]],
                self.annotation_df[[self.y_col]],
                test_size=perc,
                stratify=self.annotation_df[[self.y_col]])
        else:
            self.X_train_index, self.X_test_index, self.y_train, self.y_test = train_test_split(
                self.annotation_df[[self.x_col]],
                self.annotation_df[[self.y_col]],
                test_size=perc)
        # save all files for reproducibility
        self.X_train_index.to_csv('X_train_index.csv')
        self.X_test_index.to_csv('X_test_index.csv')
        self.y_train.to_csv('y_train.csv')
        self.y_test.to_csv('y_test.csv')


    def file_read(self, file_index, output_file_name=False):
        """
        Read a wav file from the acoustic_folder where the name of the file has an index.
        :param file_index: index of the file we need to search for
        :type: int
        :param output_file_name: boolean indicating if file name should be outputed
        :type output_file_name: bool
        :return: file name, samples and sample rate arrays
        :rtype: string, numpy.ndarray and int
        """
        if type(file_index) != np.int64:
            raise ValueError('Invalid file_index type. file_index is type %s and expected type is int.' % type(file_index).__name__)
        try:
            file_name = [x for x in self.accoustic_files if x.find('index' + str(file_index) + '.wav') != -1][0]
            file_name = self.acoustic_folder + file_name
            logging.info('%s file exists.' % file_name)
            samples, sample_rate = librosa.load(file_name, sr=None, mono=True, offset=0.0, duration=None)
            if output_file_name:
                return file_name, samples, sample_rate
            else:
                return  samples, sample_rate

        except:
            logging.info('File %s DOES NOT exist in %s' %(file_index, self.acoustic_folder))
            # raise ValueError('File %s DOES NOT exist in %s' %(file_name, self.acoustic_folder))

        try:
            file_name = [x for x in self.augmented_files if x.find('index' + str(file_index) + '.wav') != -1][0]
            file_name = self.augment_folder + file_name
            logging.info('%s file exists.' % file_name)
            samples, sample_rate = librosa.load(file_name, sr=None, mono=True, offset=0.0, duration=None)
            if output_file_name:
                return file_name, samples, sample_rate
            else:
                return  samples, sample_rate

        except:
            logging.info('File %s DOES NOT exist in %s' % (file_index, self.acoustic_folder))
            # raise ValueError('File %s DOES NOT exist in %s' % (file_name, self.augment_folder))


    def dataframe_to_dataset_split_save(self,df, split_type,file_name):
        """
        Converts a pandas dataframe to data dict which is used in hugging face transformers. Then saves the data to the datadict_folder.
        :param df: pandas data frame which has to be converted
        :type df: pd.DataFrame
        :param split_type: 'train' or 'test' for training and testing sets
        :type split_type: str
        :param file_name: name of the file to be saved
        :type file_name: str
        """
        split_type_list = ['train','test']
        if type(df) != pd.core.frame.DataFrame:
            raise ValueError('Invalid df type. df is type %s and expected type is pd.core.frame.DataFrame.' %type(df).__name__)
        if type(file_name) != str:
            raise ValueError(
                'Invalid file_name type. It is type %s and expected type is str.' % type(file_name).__name__)
        if split_type not in split_type_list:
            raise ValueError(
                'Invalid split type. It should be from the list %s' %split_type_list)
        if 'index' not in df.columns.to_list():
            raise ValueError('Column index is not part of df. It is a requirement. The index should provide the file index of the file to be read.')
        dataset = pd.DataFrame({})
        for train_index, row in df.iterrows():
            try:
                temp_dataset = pd.DataFrame({})
                path, sample, sample_rate = self.file_read(row['index'],output_file_name=True)

                temp_dataset['audio'] = [{'path':path,
                                         'array':sample,
                'sampling_rate':sample_rate}]

                # to update here to have path as an array with everything below
                temp_dataset['train_index'] = train_index
                temp_dataset['file_index'] = row['index']
                if split_type == 'train':
                    temp_dataset['label'] = self.y_train.loc[train_index, self.y_col]
                else:
                    temp_dataset['label'] = self.y_test.loc[train_index, self.y_col]
                dataset = pd.concat([dataset, temp_dataset], axis=0)
            except:
                pass
        data = Dataset.from_pandas(dataset, split=split_type)
        logging.info('Dataframe transformed to dataset.')
        #save the file with the first and the last index
        #Note: we use this function because it takes less than a sec to load the data, for .json functions, it took 3 min per chunk

        if split_type == 'train':
            data.save_to_disk(self.datadict_folder+"/train/"+file_name)
        elif split_type == 'test':
            data.save_to_disk(self.datadict_folder + "/test/" + file_name)
        else:
            raise ValueError(
                'Invalid split type. It should be from the list %s' % split_type_list)
        logging.info('Save the data set.')
    def dataframe_to_dataset(self,df, split_type, num_chunks=10):
        """
        Splits a dataframe in specific number of chunks. Converts a pandas dataframe to data dict which is used in hugging face transformers. Then saves those chunks into files, reads them and returns the train data.
        :param df: pandas data frame which has to be converted
        :type df: pd.DataFrame
        :param split_type: 'train' or 'test' for training and testing sets
        :type split_type: str
        :param num_chunks: number of chunks to split the data
        :rtype num_chunks: int
        :return: Dataset
        :rtype: Dataset
        """
        split_type_list = ['train','test']
        if type(df) != pd.core.frame.DataFrame:
            raise ValueError('Invalid df type. df is type %s and expected type is pd.core.frame.DataFrame.' %type(df).__name__)
        if type(num_chunks) != int:
            raise ValueError(
                'Invalid num_chunks type. It is type %s and expected type is int.' % type(num_chunks).__name__)
        if split_type not in split_type_list:
            raise ValueError(
                'Invalid split type. It should be from the list %s' %split_type_list)
        if 'index' not in df.columns.to_list():
            raise ValueError('Column index is not part of df. It is a requirement. The index should provide the file index of the file to be read.')
        #creating split folder
        if split_type == 'train':
            split_folder='train/'
        elif split_type == 'test':
            split_folder='test/'
        else:
            raise ValueError(
                'Invalid split type. It should be from the list %s' % split_type_list)
        #clean the data in the folder
        try:
            clean_directory(self.datadict_folder + split_folder, folder=True)
        except:
            pass

        # get all indices for the df
        all_indices = np.array_split(df.index, num_chunks)
        #intentionally we use for loop to reduce the memory usage
        for set_indices in all_indices:
            time.sleep(3)
            # save the file with the first and the last index
            file_name = "data%s_%s.hf"%(str(set_indices[0]),str(set_indices[len(set_indices)-1]))
            try:
                self.dataframe_to_dataset_split_save(df.loc[set_indices,],split_type=split_type,file_name=file_name)
            except:
                pass
        logging.info('All Dataframes transformed to dataset and saved to %s' %self.datadict_folder)

        # concatinate all sub-data into one data and save it
        # get all file names

        hf_files  = get_file_names(self.datadict_folder+split_folder)

        data = datasets.load_from_disk(self.datadict_folder+split_folder+hf_files[0])
        for f in hf_files[1:]:
            data_temp = datasets.load_from_disk(self.datadict_folder+split_folder+f)
            data = datasets.concatenate_datasets([data, data_temp])
        logging.info('All data is loaded.')
        data = data.class_encode_column("label")
        return data

    def dataframe_to_datadict(self,train_df, test_df):
        """
        Converts two data frames (test and train) into a data dict which will be used for HuggingFace transformers.
        :param train_df: data frame for the training set
        :type train_df: pd.DataFrame
        :param test_df: data frame for the testing set
        :type test_df: pd.DataFrame
        :return: datadict_data, stored in the object
        :rtype: datadict
        """
        # self.train_df_dataset = self.dataframe_to_dataset(train_df, split_type='train')
        # self.test_df_dataset = self.dataframe_to_dataset(test_df, split_type='test')
        train_df_dataset = self.dataframe_to_dataset(train_df, split_type='train')
        test_df_dataset = self.dataframe_to_dataset(test_df, split_type='test')
        datadict_data = datasets.DatasetDict(
            {
                "train": train_df_dataset,
                "test": test_df_dataset,
            }
        )
        logging.info('Data dictionary is created.')
        return datadict_data

    def data_augmentation_row(self,arg):
        """
        A row-wise function which augments acoustic data and saves it in the acoustic_folder.

        :param arg: tuple with first argument the index of each row of a data frame, second argument - the actual row of the data frame
        :type arg: tuple

        :return: list of lists with the augmented data file index, the augmented data train index, the label of the augmented file and the original train index. If an issue occurs, only the original train and file indices are returned.
        :rtype: list

        """
        if type(arg) != tuple:
            raise ValueError('Invalid arg type. arg is type %s and expected type is list.' %type(arg).__name__)
        train_index, row,n= arg

        # get the necessary indices to trace files easily
        file_index = row[self.x_col]  # this index is necessary to ensure we have the correct file name (coming from the annotation file)
        label = self.y_train.loc[train_index, self.y_col]
        # check if the file from the annotation data exists in the folders
        try:
            # read the file
            samples, sample_rate = self.file_read(file_index)
            augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TanhDistortion(min_distortion=0.01,max_distortion=0.7,p=0.5),
                GainTransition(p=0.5),
                AirAbsorption(min_distance=1.0,max_distance=10.0,p=0.5)

            ])
            augmented_samples = augment(samples=samples, sample_rate=sample_rate)
            augmented_file_index = int(str(n) + str(0) + str(train_index))
            augmented_file_name = 'index' + str(augmented_file_index) + '.wav'
            sf.write(self.augment_folder + augmented_file_name, augmented_samples, sample_rate)
            return [augmented_file_index, label, train_index]
        except:
            return  [None, None, None]
            logging.warning('File with index %s is NOT augmented' % str(file_index))

    def data_augmentation_df(self,N=1):
        """
        A function which augments the train data and saves it in the augmented folder (initially cleans the folder); replaces the train data by adding the augmented files information; stores the information in augmented_df; and saves the names of the augmented files in augmented_files.
        :param N: the number of times the augmentation process should happen
        :type N: int
        :return: the augmented data is saved
        """

        if type(N) != int:
            raise ValueError('Invalid N type. arg is type %s and expected type is int.' %type(N).__name__)

        # clean augmented directory
        clean_directory(self.augment_folder)

        logging.info('Augmented folder is cleaned.')
        self.augmented_df = pd.DataFrame()

        for n in range(1,N+1,1):
            # create the augmented files
            pool = mp.Pool(processes=mp.cpu_count())
            augmented = pool.map(self.data_augmentation_row,
                                     [(train_index, row, n) for train_index, row in self.X_train_index.iterrows()])
            augmented_df = pd.DataFrame(augmented, columns=['augmented_file_index', 'label', 'train_index'])
            augmented_df.dropna(inplace=True)
            augmented_df['augmented_file_index'] = augmented_df['augmented_file_index'].astype(int)
            augmented_df['train_index'] = augmented_df['train_index'].astype(int)
            augmented_df.index = augmented_df['augmented_file_index']
            # augmented_df.dropna(inplace=True)
            self.augmented_df = pd.concat([self.augmented_df,augmented_df])

            logging.info('Augmented data is created.')
            # split X
            X_train_index = pd.DataFrame(augmented_df['augmented_file_index'])
            X_train_index.columns = self.X_train_index.columns
            # add the augmented data to the existing data
            self.X_train_index = pd.concat([X_train_index, self.X_train_index])

            logging.info('X_train index is updated.')
            # split y
            y_train = pd.DataFrame(augmented_df['label'])
            y_train.columns = [self.y_train.columns[0]]
            # add the augmented data to the existing data
            self.y_train = pd.concat([self.y_train,y_train])
            logging.info('Y_train is updated.')

        #save the names of the augmented files for easy access later on
        self.augmented_files = get_file_names(self.augment_folder)


    def data_transformation_row(self, arg):
        """
        A row-wise function which finds the correct file from the annotation data frame and then transforms the acoustic data with mfcc or mel spec. n_fft is set to 1000 due to the nature of the audio data.
        :param arg: tuple with first argument the index of each row of a data frame, second argument - the actual row of the data frame and third argument - data frame with the dependant variable
        :type arg: tuple

        :return: list of lists with the transformed data. The non-existent files are returned as None type.
        :rtype: list
        """

        if type(arg) != tuple:
            raise ValueError('Invalid arg type. arg is type %s and expected type is list.' %type(arg).__name__)
        train_index, row, func = arg

        # get the necessary indices to trace files easily
        file_index = row[self.x_col]  # this index is necessary to ensure we have the correct file name (coming from the annotation file)
        # check if the file from the annotation data exists in the folders
        try:
            # read the file
            samples, sample_rate = self.file_read(file_index)
            # check if the data extraction is correct through the duration of the sample
            duration_of_sound = round(len(samples) / sample_rate, 2)
            annotation_duration = self.annotation_df.loc[self.annotation_df['index'] == file_index, 'duration']
            if duration_of_sound == round(annotation_duration.loc[train_index,], 2):
                logging.info('File with index %s has the correct duration.' %str(file_index))
            else:
                logging.warning('File with index %s DOES NOT have the correct duration.' %str(file_index))

            if func == 'mfcc':
                sample_transformed = np.mean(librosa.feature.mfcc(y=samples, sr=sample_rate,
                                                                  n_mfcc=100, n_fft = 1000).T, axis=0)
                sample_transformed = np.insert(sample_transformed, 0, train_index)
            elif func == 'mel spec':
                try:
                    mel = librosa.feature.melspectrogram(y = samples, sr=sample_rate, n_fft=1000,
                                                         hop_length=512, n_mels=128)
                    sample_transformed = librosa.power_to_db(mel)
                    sample_transformed = sample_transformed.reshape(1,-1)
                    sample_transformed = sample_transformed[0]
                # we need to add the indices for tracking purposes
                except:
                    sample_transformed= np.zeros()

            logging.info('File with index %s is transformed and added to the transformed data frame' %str(file_index))

            # return the transformed file and add the indices
            return sample_transformed

        except:
            return [train_index,file_index]
            logging.warning('File with index %s is NOT added to the transformed data frame' %str(file_index))



    def data_transformation_df(self, X, func):
        """
        Find the correct file from the annotation data frame and then transform the acoustic data using mfcc or mel spec methods. Store the index from the annotation data frame (the key) and the df index to track the associated y values.
        :param X: pandas data frame with the indices of the acoustic files which need to be transformed
        :type X: pandas.DataFrame
        :param func: function for audio files transformation. One can choose from a list of options ['mfcc','mel spec']
        :type func: function or method

        :return: data frame with the transformed data
        :rtype:pandas.dataFrame
        """
        func_list = ['mfcc','mel spec']
        if type(X) != pd.core.frame.DataFrame:
            raise ValueError('Invalid arg type. arg is type %s and expected type is pandas.core.frame.DataFrame.' % type(X).__name__)
        if 'index' not in X.columns.to_list():
            raise ValueError('Column index is not part of X data frame. It is a requirement.')
        if func not in func_list:
            raise ValueError(
                'Invalid function. function should be from the list %s' %func_list)


        pool = mp.Pool(processes=mp.cpu_count())
        X_transformed = pool.map(self.data_transformation_row,[(train_index, row, func) for train_index, row in X.iterrows()])
        # add the column names
        cols = ['train_index','file_index']
        max_length = max([len(x) for x in X_transformed if x is not None])
        cols = cols + ['col' + str(x) for x in range(max_length - 2)]
        # transform to data frame
        X_df = pd.DataFrame(columns=cols)
        for x in X_transformed:
            if len(x) == max_length:
                X_df.loc[len(X_df)] = x
            else:
                x_updated = list(x)+list([1]*(max_length-len(x)))
                X_df.loc[len(X_df)] = x_updated
        logging.info('Whole data frame transformed.')
        return X_df


    def transformer_classification(self
                                   , data
                                   , max_duration=5
                                   , model_id ='facebook/hubert-base-ls960'
                                    ,batch_size = 8
                                    ,gradient_accumulation_steps = 4
                                    ,num_train_epochs = 10
                                   ,warmup_ratio=0.1
                                   ,logging_steps = 10
                                   ,learning_rate = 3e-5
                                   ,name='finetuned-bee'
                                    ):
        """Execute huggingface transformer pre-trained classification model for audio data
        :param data: DataDict for audio data with train and test
        :type data: DataDict
        :param max_duration: maximum duration of the data file
        :type max_duration: int
        :model_id: the name of the HiggingFace model
        :type model_id: str
        :param batch_size: The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training/testing.
        :type batch_size: int
        :param gradient_accumulation_steps: Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        :type gradient_accumulation_steps: int
        :param num_train_epochs:Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).
        :type num_train_epochs: float
        :param warmup_ratio: Ratio of total training steps used for a linear warmup from 0 to learning_rate.
        :type warmup_ratio: float
        :param logging_steps:Number of update steps between two logs if logging_strategy="steps". Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps.
        :type logging_steps: float
        :param learning_rate:  The initial learning rate.
        :type learning_rate: float
        :param name: name of the newly created model
        :type name: str
        :return: trained model
        :rtype: HuggingFace.models
        """
        #checks for the correct inputs
        if type(data) != datasets.dataset_dict.DatasetDict:
            raise ValueError('Invalid arg type. arg is type %s and expected type is datasets.dataset_dict.DatasetDict.' % type(data).__name__)
        if type(model_id) != str:
            raise ValueError('Invalid arg type. arg is type %s and expected type is str.' % type(model_id).__name__)
        if type(name) != str:
            raise ValueError('Invalid arg type. arg is type %s and expected type is str.' % type(name).__name__)
        if type(batch_size) != int:
            raise ValueError('Invalid arg type. arg is type %s and expected type is int.' % type(batch_size).__name__)
        if type(max_duration) != int:
            raise ValueError('Invalid arg type. arg is type %s and expected type is int.' % type(max_duration).__name__)
        if type(gradient_accumulation_steps) != int:
            raise ValueError('Invalid arg type. arg is type %s and expected type is int.' % type(gradient_accumulation_steps).__name__)
        if type(num_train_epochs) != float:
            raise ValueError('Invalid arg type. arg is type %s and expected type is float.' % type(num_train_epochs).__name__)
        if type(warmup_ratio) != float:
            raise ValueError('Invalid arg type. arg is type %s and expected type is float.' % type(warmup_ratio).__name__)
        if type(logging_steps) != float:
            raise ValueError('Invalid arg type. arg is type %s and expected type is float.' % type(logging_steps).__name__)
        if type(learning_rate) != float:
            raise ValueError('Invalid arg type. arg is type %s and expected type is float.' % type(learning_rate).__name__)
        #create the feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_id, do_normalize=True, return_attention_mask=True )
        # resample the data to have the same sampling rate as the pretrained model
        sampling_rate = feature_extractor.sampling_rate
        data = data.cast_column("audio", Audio(sampling_rate=sampling_rate))
        logging.info('Data is transformed to the required sampling rate.')

        #create numeric labels
        id2label_fn = data["train"].features[self.bee_col].int2str

        id2label_fn = data["train"].features[self.bee_col].int2str
        id2label = {
            str(i): id2label_fn(i)
            for i in range(len(data["train"].features[self.bee_col].names))
        }
        label2id = {v: k for k, v in id2label.items()}

        num_labels = len(id2label)
        logging.info('Labels are constructed')

        #construct the model
        model = AutoModelForAudioClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        logging.warning('Model is initiated.')
        #add the train arguments
        model_name = model_id.split("/")[-1]

        training_args = TrainingArguments(
            "%s-%s" %(model_name,name),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            fp16=True,
            gradient_checkpointing=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )
        logging.info('Training arguments are set.')
        #encode the data
        data_encoded = data.map(
            preprocess_function,
            remove_columns=["audio", "file_index"],
            batched=True,
            batch_size=100,
            num_proc=1,
            fn_kwargs={"feature_extractor": feature_extractor, "max_duration": max_duration}
        )
        logging.info('Data is encoded with the feature extractor.')
        trainer = Trainer(
            model,
            training_args,
            train_dataset=data_encoded["train"],
            eval_dataset=data_encoded["test"],
            tokenizer=feature_extractor,
            compute_metrics=compute_metrics
        )

        trainer.train()
        logging.info('Model is trained.')

        return trainer

    def best_model(self, model, param_dist):
        """Identify the best model after tuning the hyperparameters

        :param model: an initiated machine learning model
        :type model: Any
        :param param_dist: a dictionary with the parameters and their respective ranges for the tuning
        :type param_dist: dict
        :return: RandomizedSearchCV object
        :rtype: RandomizedSearchCV
        """

        if type(param_dist) != dict:
            raise ValueError(
                'Invalid arg type. arg is type %s and expected type is dict.' % type(param_dist).__name__)

        try:
            best_model = RandomizedSearchCV(model,
                                            param_distributions=param_dist,
                                            n_iter=100,
                                            cv=10)
            logging.info('Parameter tuning completed')
        except Exception as error:
            logging.error(error)
        return best_model

    def accuracy_metrics(self, y_pred):
        """ Provide accuracy metrics to compare the different models

        :param y_pred: predicted dependent values
        :type y_pred: list
        :return: accuracy, precision, recall
        :rtype: tuple
        """
        if type(y_pred) != np.ndarray:
            raise ValueError(
                'Invalid arg type. arg is type %s and expected type is np.ndarray.' % type(y_pred).__name__)
        try:
            # accuracy score
            acc = accuracy_score(self.y_test, y_pred)
            # calculate the precision score
            precision = precision_score(self.y_test, y_pred, average='macro')
            recall = recall_score(self.y_test, y_pred, average='macro')

            logging.info('Accuracy metrics calculated')
        except Exception as error:
            logging.error(error)

        return acc, precision, recall

    def misclassified_analysis(self, y_pred):
        """Misclassification analysis to understand where the model miscalculates and if any pattern can be found

        :param y_pred: predicted dependent values
        :type y_pred: list
        :return: misclassified values
        :rtype: pandas.Series
        """
        if type(y_pred) != np.ndarray:
            raise ValueError(
                'Invalid arg type. arg is type %s and expected type is np.ndarray.' % type(y_pred).__name__)
        try:

            # check the misclassified datapoints
            y_test_df = pd.DataFrame(self.y_test)
            y_test_df['pred'] = y_pred
            y_test_df['check'] = y_test_df[self.y_col] == y_test_df['pred']
            misclassified = y_test_df.loc[y_test_df[~y_test_df['check']].index,:].value_counts()
            logging.info('Misclassified analysis completed')
        except Exception as error:
            logging.error(error)

        return misclassified

    def model_results(self
                      , model
                      , param_dist
                      ,func='mfcc'
                      ,do_pca=True):
        """Provide a full picture of the model performance and accuracy
        :param model: an initiated machine learning model such as Random Forest
        :type model: Any
        :param param_dist: a dictionary with the parameters and their respective ranges for the tuning
        :type param_dist: dict
        :param func:function to transform the input variables. Possible values are 'mfcc' and 'mel spec'
        :type func: str
        :param do_pca: whether to run pca on the data and take the first two dimensions
        :type do_pca: bool
        :return: the best model with its accuracy metrics, misclassified analysis and pca explained variance (do_pca = True)
        :rtype: tuple
        """
        if type(param_dist) != dict:
            raise ValueError(
                'Invalid arg type. arg is type %s and expected type is dict.' % type(param_dist).__name__)
        if type(do_pca) != bool:
            raise ValueError(
                'Invalid arg type. arg is type %s and expected type is dict.' % type(do_pca).__name__)
        if func not in ['mfcc', 'mel spec']:
            raise ValueError(
                'Invalid arg type. arg is expected to be either mfcc or mel spec but it is' % func)

        try:
            # Use random search to find the best hyperparameters
            rand_search = self.best_model(model=model, param_dist=param_dist)

            # transform the X_train and then get only the correct entries
            self.X_train = self.data_transformation_df(self.X_train_index, func=func)

            self.X_test = self.data_transformation_df(self.X_test_index, func = func)

            # standardise the values
            x = self.X_train[[x for x in self.X_train.columns if x not in ['train_index', 'file_index']]]
            y = np.array(self.y_train).ravel()
            x = StandardScaler().fit_transform(x)
            x_test = self.X_test[[x for x in self.X_test.columns if x not in ['train_index', 'file_index']]]
            x_test = StandardScaler().fit_transform(x_test)

            if do_pca:

                #conduct PCA to reduce overfitting

                pca = PCA(n_components=2)
                principalComponents = pca.fit_transform(x)
                pca_variance = pca.explained_variance_ratio_
                principalDf = pd.DataFrame(data=principalComponents
                                           , columns=['principal component 1', 'principal component 2'])
                principalComponents_test = pca.transform(x_test)
                principalDf_test = pd.DataFrame(data=principalComponents_test
                                                , columns=['principal component 1', 'principal component 2'])

                # fit the best model
                rand_search.fit(principalDf,y)

                # generate predictions with the best model
                y_pred = rand_search.predict(principalDf_test)

                # calculate model accuracy
                acc, precision, recall = self.accuracy_metrics(y_pred=y_pred)

                # check the misclassified datapoints
                misclassified = self.misclassified_analysis(y_pred=y_pred)


                logging.info('Model Results Calculated')
                return acc, precision, recall, rand_search, misclassified, pca_variance
            else:
                # fit the best model
                rand_search.fit(x, y)

                # generate predictions with the best model
                y_pred = rand_search.predict(x_test)

                # calculate model accuracy
                acc, precision, recall = self.accuracy_metrics(y_pred=y_pred)

                # check the misclassified datapoints
                misclassified = self.misclassified_analysis(y_pred=y_pred)

                logging.info('Model Results Calculated')
                return acc, precision, recall, rand_search, misclassified
        except Exception as error:
            logging.error(error)




    def random_forest_results(self, func, do_pca = True):
        """Run Random Forest and conduct hyperparameter tuning, accuracy measurement and feature importance
        :param func:function to transform the input variables. Possible values are 'mfcc' and 'mel spec'
        :type func: str
        :param do_pca: whether to run pca on the data and take the first two dimensions
        :type do_pca: bool
        :return: accuracy, precision, recall, misclassified analysis, pca variance (do_pca=True) and feature importance
        :rtype: tuple
        """
        if func not in ['mfcc', 'mel spec']:
            raise ValueError(
                'Invalid arg type. arg is expected to be either mfcc or mel spec but it is' % func)
        if type(do_pca) != bool:
            raise ValueError(
                'Invalid arg type. arg is type %s and expected type is dict.' % type(do_pca).__name__)
        try:
            param_dist = {'n_estimators': [20, 30, 40],
                          'max_depth': [2, 8, 10, 12, 14, 16, 18, 20]}
            # Create a random forest classifier
            rf = RandomForestClassifier()

            # check the model results
            if do_pca:
                acc, precision, recall,  rand_search,misclassified, pca_variance = self.model_results(model=rf
                                                                                                      , param_dist=param_dist
                                                                                                      ,do_pca=True)
            else:
                acc, precision, recall, rand_search, misclassified = self.model_results(model=rf ,param_dist=param_dist
                                                                                                      ,do_pca=False)
            logging.info('Accuracy metrics are calculated.')
            # # check the features importance
            best_model = rand_search.best_estimator_
            importances = best_model.feature_importances_
            forest_importances = pd.Series(importances, index=[x for x in self.X_train.columns if x not in ['train_index', 'file_index']])
            logging.info('Importance is calculated.')
            if do_pca:
                return acc, precision, recall , misclassified, pca_variance, forest_importances
            else:
                return acc, precision, recall, misclassified, forest_importances
        except Exception as error:
            logging.error(error)
