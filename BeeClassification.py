# libraries
import pandas as pd
import pyarrow as pa
from sklearn.model_selection import train_test_split,RandomizedSearchCV
import logging
import os
import numpy as np
from scipy.fft import fft,rfftfreq, fftfreq
import librosa
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import soundfile as sf
import random
import seaborn as sns
from datasets import Dataset, Audio, ClassLabel
import datasets
import time
import shutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Bee:
    """BeeNotBee class ....
    # TODO: update the documentation
    :param annotation_path: path to the annotation data
    :type annotation_path: str
    :param y_col: column label for the dependent variable in the annotation file
    :type y_col: str
    :param x_col: column label for the independent variable in the annotation file. It is used to read the wav file.
    :type x_col: str
    :param acoustic_folder: folder name where acoustic files are stored
    :type acoustic_folder: str




    :param logname: path to the log file
    :type logname: str

    :return: BeeNotBee object
    :rtype: BeeNotBee
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
        self.annotation_path =annotation_path
        self.annotation_dtypes_path = annotation_dtypes_path
        self.x_col = x_col
        self.y_col = y_col
        self.bee_col = bee_col
        self.acoustic_folder = acoustic_folder
        self.augment_folder = augment_folder
        self.datadict_folder = datadict_folder
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)
    def plot_figure(self
                    , plot_title
                    , file_title
                    , plot_code):
        """ Plotting a figure based on dynamic code

        :param plot_title: plot title
        :type plot_title: str
        :param file_title: title of the saved file
        :type file_title: str
        :param plot_code: code to be executed for the plot to visualize
        :type plot_code: str
        :return: saved file with the requested plot
        """
        try:
            plt.figure(figsize=(16, 6))
            exec(plot_code)
            plt.title(plot_title)
            plt.savefig(file_title, dpi=300, bbox_inches='tight')
            logging.info("Graph is plotted")
        except Exception as error:
            logging.error(error)


    def read_annotation_csv(self):
        """
        Read annotation data
        :return: pandas data frame with the annotations
        :rtype: pandas.DataFrame
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
        :rtype: pandas.DataFrame
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
        :rtype: log
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


    def split_annotation_data(self,perc=0.25, no_bee = False, data_quality = True, stratified = True):
        """
        Split the annotation data into train and test based on the y_col and x_col values. Only bee or bee and non-bee files can be used. Data can be enhanced as well. Save the csv files.
        :param perc: test split percentage, a number between 0.0 and 1.0
        :type perc: float
        :param no_bee: Boolean value to indicate if no_bee files to be added in the data set.
        :type no_bee: bool
        :param data_quality: Boolean value to indicate if only the quality data should be kept. This means files which exist in the directory and are above 2 sec. long.
        :type data_quality: bool
        :param stratified: Boolean value to indicate if the split should be stratified
        :type stratified: bool
        :return: X train, X test, y train and y test pandas data frames
        :rtype: pandas.DataFrame
        """
        if type(perc) != float:
            raise ValueError(
                'perc input is not the correct type. It is type %s, but it should be a float.' % type(perc))
        if perc < 0.0:
            raise ValueError('perc should be bigger than 0.')
        if perc > 1.0:
            raise ValueError('perc should be smaller than 1.')
        if type(no_bee) != bool:
            raise ValueError('no_bee input is not the correct type. It is type %s, but it should be a boolean.' % type(no_bee))
        if type(data_quality) != bool:
            raise ValueError('stratified input is not the correct type. It is type %s, but it should be a boolean.' % type(data_quality))
        if type(stratified) != bool:
            raise ValueError(
                'stratified input is not the correct type. It is type %s, but it should be a boolean.' % type(
                    stratified))
        if self.x_col not in self.annotation_df:
            raise ValueError('x_col not in the annotation_df. Change x_col.')
        if self.y_col not in self.annotation_df:
            raise ValueError('y_col not in the annotation_df. Change y_col.')
        else:

            if data_quality:
                existing_indices = pd.DataFrame()
                existing_indices['index'] = [int(f.split('index')[1].split('.wav')[0]) for f in self.accoustic_files]
                existing_indices['Dir Exist'] = True

                self.annotation_df = self.annotation_df.merge(existing_indices, how='left', left_on='index', right_on='index')
                self.annotation_df['Dir Exist'].fillna(False, inplace=True)
                # 367 files are missing
                #here we may need to update something in future so that it is more universal
                annotation_df_updated  = self.annotation_df[self.annotation_df['Dir Exist']]
                annotation_df_updated = annotation_df_updated[annotation_df_updated['duration']>2.0]
            else:
                annotation_df_updated = self.annotation_df
            #TODO what is this doing?
            # maybe this was in case if we want to model the bee-nobee data? need to check this
            if not no_bee:
                annotation_df_updated = annotation_df_updated[annotation_df_updated[self.bee_col]=='bee']

            if stratified:
                self.X_train_index, self.X_test_index, self.y_train, self.y_test = train_test_split(
                    annotation_df_updated[[self.x_col]],
                    annotation_df_updated[[self.y_col]],
                    test_size=perc,
                    stratify=annotation_df_updated[[self.y_col]])
            else:
                self.X_train_index, self.X_test_index, self.y_train, self.y_test = train_test_split(
                    annotation_df_updated[[self.x_col]],
                    annotation_df_updated[[self.y_col]],
                    test_size=perc)

            # save all files for reproducibility
            self.X_train_index.to_csv('X_train_index.csv')
            self.X_test_index.to_csv('X_test_index.csv')
            self.y_train.to_csv('y_train.csv')
            self.y_test.to_csv('y_test.csv')


    def acoustic_file_names(self):
        """
        TODO abstract so that we can read the name of the augmented files as well
        Create a list of files which have bee and no-bee data
        :return: a list - acoustic_files
        :rtype: list
        """
        self.accoustic_files = os.listdir(self.acoustic_folder)
        logging.info('Files in the two directories are stores in a list - acoustic_files')
        if len(self.accoustic_files)==0:
            raise ValueError('bee_files list is empty. Please, check the %s folder' % self.acoustic_folder)


    def harley_transformation_with_window(self,x,window = np.hanning):
        """
        Transform np.array to fast harley transformation with a window function
        :param x: time sequence
        :type x: numpy array
        :param window: windowing function from the list np.hanning, np.bartlett, np.blackman, np.hamming. More information for the windows functions here https://numpy.org/doc/stable/reference/routines.window.html
        :type window: numpy function
        :return:
        """


        window_list = [np.hanning, np.bartlett, np.blackman, np.hamming]
        if window not in window_list:
            raise ValueError('Invalid window type. Expected one of: %s' %['np.'+w.__name__ for w in window_list])
        elif type(x) != np.ndarray:
            raise ValueError('Invalid x type. x is type %s and expected type is np.array.' %type(x).__name__)
        else:
            x_trans = fft(x * window(len(x)))  # adding window function
            x_trans = np.real(x_trans) - np.imag(x_trans)
            logging.info('Harley transformation is completed for the array conducted with window function %s' %window.__name__ )
        return x_trans

    def freq_powerspect_func(self,x, dt, npnts, denoise=True, cutoff=0):
        """
        Transform the FFT harley vector into frequency and power vectors
        :param x: time sequence
        :type x: numpy array
        :param t: time vector
        :type t: numpy array
        :param dt: sampling interval
        :type dt: list
        :param npnts: number of time points
        :type npnts:  int
        :param denoise: filter out noise
        :type denoise: bool
        :param cutoff: denoise below the cutoff amplitude
        :type cutoff: int
        :return: frequency list and power specter list
        :rtype: list
        """
        if type(x) != np.ndarray:
            raise ValueError('Invalid x type. x is type %s and expected type is np.array.' %type(x).__name__)
        elif type(npnts) != int:
            raise ValueError('Invalid npnts type. npnts is type %s and expected type is int.' %type(npnts).__name__)
        elif type(dt) != float:
            raise ValueError('Invalid dt type. dt is type %s and expected type is float.' %type(dt).__name__)
        elif type(denoise) != bool:
            raise ValueError('Invalid denoise type. denoise is type %s and expected type is bool.' %type(denoise).__name__)
        elif type(cutoff) != int:
            raise ValueError('Invalid cutoff type. cutoff is type %s and expected type is int.' %type(cutoff).__name__)
        else:

            # TODO Maybe we can use this piece of code instead https://huggingface.co/learn/audio-course/en/chapter1/audio_data
            # # get the amplitude spectrum in decibels
            # amplitude = np.abs(dft)
            # amplitude_db = librosa.amplitude_to_db(amplitude, ref=np.max)



            X = self.harley_transformation_with_window(x)
            #n = np.arange(npnts)
            #T = npnts * dt
            freq = rfftfreq(npnts, dt)
            #freq = n / T
            powerspect = 2 * np.abs(X) / npnts
            if denoise:
                powerspect = powerspect * (powerspect > cutoff)  # Zero all frequencies with small power
                X = X * (powerspect > cutoff)  # Zero small Fourier coefficients
            logging.info('Frequency and power specter is calculated for the time vector')
            return freq, powerspect


    def binning(self, x, dt, npnts, n_bins =128, n_start = 0.0 , n_end = 2000.0, denoise=True, cutoff=0):
        #TODO - we may not need this any more
        """
        Transform a time to frequency using the freq_powerspect_func and then binning it to a specific number of frequency vectors. The power in every bin is the average of the original frequency vector.
        :param x: time sequence
        :type x: numpy array
        :param dt: sampling interval
        :type dt: list
        :param npnts: number of time points
        :type npnts:  int
        :param n_bins: number of bins in the new array
        :type n_bins: int
        :param n_start: from where the binning to start
        :type n_start: float
        :param n_end: until where the binning to stop
        :type n_end: float
        :param denoise: filter out noise
        :type denoise: bool
        :param cutoff: denoise below the cutoff amplitude
        :type cutoff: int
        :return: list of the binned inputs
        :rtype: list
        """
        if type(x) != np.ndarray:
            raise ValueError('Invalid x type. x is type %s and expected type is np.array.' %type(x).__name__)
        elif type(npnts) != int:
            raise ValueError('Invalid npnts type. npnts is type %s and expected type is int.' %type(npnts).__name__)
        elif type(dt) != float:
            raise ValueError('Invalid dt type. dt is type %s and expected type is float.' %type(dt).__name__)
        elif type(denoise) != bool:
            raise ValueError('Invalid denoise type. denoise is type %s and expected type is bool.' %type(denoise).__name__)
        elif type(cutoff) != int:
            raise ValueError('Invalid cutoff type. cutoff is type %s and expected type is int.' %type(cutoff).__name__)
        elif type(n_bins) != int:
            raise ValueError('Invalid n_bins type. n_bins is type %s and expected type is int.' % type(n_bins).__name__)
        elif type(n_start) != float:
            raise ValueError('Invalid n_start type. n_start is type %s and expected type is float.' % type(n_start).__name__)
        elif type(n_end) != float:
            raise ValueError('Invalid n_end type. n_end is type %s and expected type is float.' % type(n_end).__name__)
        else:
            # to ensure the length of the bins is unchanged, add 1
            n_bins=n_bins+1
            freq, powerspect = self.freq_powerspect_func(x=x,npnts=npnts, dt=dt,denoise=denoise,cutoff=cutoff)
            bins = np.linspace(n_start, n_end, n_bins)
            bins_list = [(bins[i], bins[i + 1]) for i in (range(len(bins) - 1))]
            binned_x = list()
            for pair in bins_list:
                b0 = pair[0]
                b1 = pair[1]
                index = ([b0] + [i for i, e in enumerate(freq) if e < b1])[-1]
                try:
                    old_index
                except NameError:
                    old_index = 0
                try:
                    p = sum(powerspect[old_index:index]) / len(powerspect[old_index:index])
                except:
                    p = 0
                binned_x.append(p)
                old_index = index
            logging.info('Frequency vector binned.')
            return binned_x
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
            file_name = self.acoustic_folder + file_name #TODO what about the augmented data? is the data folder ok? or it should be an input
            logging.info('%s file exists.' % file_name)
            samples, sample_rate = librosa.load(file_name, sr=None, mono=True, offset=0.0, duration=None)
            if output_file_name:
                return file_name, samples, sample_rate
            else:
                return  samples, sample_rate

        except:
            raise ValueError('File %s DOES NOT exist in %s' %(file_name, self.acoustic_folder))

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
            raise ValueError('File %s DOES NOT exist in %s' % (file_name, self.augment_folder))
    # def map_label2id(self,ClassLabels,example, col):
    #     """"TODO update the description here
    #     """
    #     example[col] = ClassLabels.str2int(example[col])
    #     return example

    def dataframe_to_dataset_split_save(self,df, split_type,file_name):
        """
        Converts a pandas dataframe to data dict which is used in hugging face transformers. Then saves the data to %s
        :param df: pandas data frame which has to be converted
        :type df: pd.DataFrame
        :param split_type: 'train' or 'test' for training and testing sets
        :type split_type: str
        :param file_name: name of the file to be saved
        :type file_name: str
        """%self.datadict_folder
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

        data = Dataset.from_pandas(dataset, split=split_type)
        data = data.class_encode_column("label")
        logging.info('Dataframe transformed to dataset.')
        #save the file with the first and the last index
        #Note: we use this function because it takes less than a sec to load the data, for .json functions, it took 3 min per chunk

        #Create label and batches
        # labels = dataset['label'].unique().tolist()
        # ClassLabels = ClassLabel(num_classes=len(labels),names=labels)
        # data = data.map(self.map_label2id, batched=True)
        # data = data.cast_column('label',ClassLabels)

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
        TODO test the raise error part for the list
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
            files_list = os.listdir(self.datadict_folder + split_folder)
            for item in files_list:
                shutil.rmtree(os.path.join(self.datadict_folder+split_folder, item))
        except:
            pass

        # get all indices for the df
        all_indices = np.array_split(df.index, num_chunks)
        #intentionally we use for loop to reduce the memory usage
        for set_indices in all_indices:
            time.sleep(3)
            # save the file with the first and the last index
            file_name = "data%s_%s.hf"%(str(set_indices[0]),str(set_indices[len(set_indices)-1]))
            self.dataframe_to_dataset_split_save(df.loc[set_indices,],split_type=split_type,file_name=file_name)
        logging.info('All Dataframes transformed to dataset and saved to %s' %self.datadict_folder)

        # concatinate all sub-data into one data and save it
        # get all file names

        hf_files  = os.listdir(self.datadict_folder+split_folder)

        data = datasets.load_from_disk(self.datadict_folder+split_folder+hf_files[0])
        for f in hf_files[1:]:
            data_temp = datasets.load_from_disk(self.datadict_folder+split_folder+f)
            data = datasets.concatenate_datasets([data, data_temp])
        logging.info('All data is loaded.')
        return data

    def dataframe_to_datadict(self,train_df, test_df):
        """
        Converts two data frames (test and train) into a data dict which will be used for HuggingFace transformers.
        :param train_df: data frame for the training set
        :type train_df: pd.DataFrame
        :param test_df: data frame for the testing set
        :type test_df: pd.DataFrame
        :return: tuple of data dict
        :rtype: Dataset
        """
        train_df_dataset = self.dataframe_to_dataset(train_df, split_type='train')
        test_df_dataset = self.dataframe_to_dataset(test_df, split_type='test')
        return train_df_dataset, test_df_dataset

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
        train_index, row = arg

        # get the necessary indices to trace files easily
        file_index = row[self.x_col]  # this index is necessary to ensure we have the correct file name (coming from the annotation file)
        label = self.y_train.loc[train_index, self.y_col]
        # check if the file from the annotation data exists in the folders
        try:
            # read the file
            samples, sample_rate = self.file_read(file_index)
            # TODO update the parameters here
            augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
            ])
            augmented_samples = augment(samples=samples, sample_rate=sample_rate)
            rand_index = random.randint(max(self.X_train_index['index'])*10,max(self.X_train_index['index'])*100)
            augmented_file_index = file_index + rand_index
            augmented_file_name = 'index' + str(augmented_file_index) + '.wav'
            sf.write(self.augment_folder + augmented_file_name, augmented_samples, sample_rate)
            return [augmented_file_index, rand_index, label, train_index]
        except:
            return [train_index, file_index]
            logging.warning('File with index %s is NOT augmented' % str(file_index))

    def data_augmentation_df(self,N=3):
        """
        A function which augments the train data and saves it in the augmented folder (initially cleans the folder); replaces the train data by adding the augmented files information; stores the information in augmented_df; and saves the names of the augmented files in augmented_files.
        :param N: the number of times the augmentation process should happen
        :type N: int
        # """

        if type(N) != int:
            raise ValueError('Invalid N type. arg is type %s and expected type is int.' %type(N).__name__)

        # clean augmented directory
        test = os.listdir(self.augment_folder)

        for item in test:
            if item.endswith(".wav"):
                os.remove(os.path.join(self.augment_folder, item))
        logging.info('Augmented folder is cleaned.')
        self.augmented_df = pd.DataFrame()
        for n in range(N):
            # create the augmented files
            pool = mp.Pool(processes=mp.cpu_count())
            augmented = pool.map(self.data_augmentation_row,
                                     [(train_index, row) for train_index, row in self.X_train_index.iterrows()])
            augmented_df = pd.DataFrame(augmented, columns=['augmented_file_index', 'rand_index', 'label', 'train_index'])
            augmented_df.index = augmented_df['rand_index']
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
            # add the augmented data to the existing data
            self.y_train = pd.concat([self.y_train,y_train])
            logging.info('Y_train is updated.')

        #save the names of the augmented files for easy access later on
        self.augmented_files = os.listdir(self.augment_folder)
        logging.info('Files in the augmented directory are stores in a list - augmneted_files')
        if len(self.augmented_files)==0:
            raise ValueError('bee_files list is empty. Please, check the %s folder' % self.augmented_files)


    def data_transformation_row(self, arg):
        """
        A row-wise function which finds the correct file from the annotation data frame and then transforms the acoustic data to binned harley fft vector. Stores the index from the annotation data frame (the key) and the df index to track the associated y values.
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

            if func == 'binning':
                # transform the file
                dt = 1 / sample_rate
                t = np.arange(0, duration_of_sound, dt)
                npnts = len(t)
                sample_transformed = self.binning(x=samples, dt=dt, npnts=npnts)
                sample_transformed = np.insert(sample_transformed, 0, file_index)
                sample_transformed = np.insert(sample_transformed, 0, train_index)
            elif func == 'mfcc':
                # TODO change the mean value to something else
                # TODO change the n_mfcc to something else
                sample_transformed = np.mean(librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=100).T, axis=0)
            elif func == 'mel spec':
                # TODO change the mean value to something else
                #TODO change the fixed values
                try:
                    mel = librosa.feature.melspectrogram(y = samples, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=128)
                    sample_transformed = mel
                    # sample_transformed = librosa.power_to_db(mel) #TODO update here to remove the power to db potentially
                    # sample_transformed = sample_transformed.reshape(1,-1)
                    # sample_transformed = sample_transformed[0][:400]#TODO update here
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
        Find the correct file from the annotation data frame and then transform the acoustic data using binning, mfcc or mel spec methods. Store the index from the annotation data frame (the key) and the df index to track the associated y values.
        :param X: pandas data frame with the indices of the acoustic files which need to be transformed
        :type X: pandas.DataFrame
        :param func: function for audio files transformation. One can choose from a list of options ['binning', 'mfcc','mel spec']
        :type func: function or method

        :return: data frame with the transformed data
        :rtype:pandas.dataFrame
        """
        func_list = ['binning', 'mfcc','mel spec']
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
                x_updated = list(x)+list([1]*(max_length-len(x))) #TODO update here the 1
                X_df.loc[len(X_df)] = x_updated
        logging.info('Whole data frame transformed.')
        return X_df


    def best_model(self, model, param_dist):
        """Identify the best model after tuning the hyperparameters

        :param model: an initiated machine learning model
        :type model: Any
        :param param_dist: a dictionary with the parameters and their respective ranges for the tuning
        :type param_dist: dict
        :return: RandomizedSearchCV object
        :rtype: RandomizedSearchCV
        """
        try:
            best_model = RandomizedSearchCV(model,
                                            param_distributions=param_dist,
                                            n_iter=100,
                                            cv=10)
            logging.info('Parameter tuning completed')
        except Exception as error:
            logging.error(error)
        return best_model

    def accuracy_metrics(self, y_pred, cm_title, cm_file_name):
        """ Provide accuracy metrics to compare the different models

        :param y_pred: predicted dependent values
        :type y_pred: list
        :param cm_title: title for the confusion matrix plot
        :type cm_title: str
        :param cm_file_name: title for the confusion matrix plot
        :type cm_file_name: str
        :return: accuracy, precision, recall and saved graph for the confusion matrix
        :rtype: list
        """
        try:
            # accuracy score
            acc = accuracy_score(self.y_test, y_pred)
            # calculate the precision score
            precision = precision_score(self.y_test, y_pred, average='macro')
            recall = recall_score(self.y_test, y_pred, average='macro')
            # confusion matrix

            self.cm = confusion_matrix(self.y_test, y_pred)

            code_str = "sns.heatmap(self.cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r')"

            self.plot_figure(
                plot_title=cm_title
                , file_title=cm_file_name
                , plot_code=code_str
            )
            logging.info('Accuracy metrics calculated')
        except Exception as error:
            logging.error(error)

        return acc, precision, recall

    def misclassified_analysis(self, y_pred):
        """Misclassification analysis to understand where the model miscalculates and if any pattern can be found

        :param y_pred: predicted dependent values
        :type y_pred: list
        :return: misclassified values
        :rtype: list
        """
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
                      , cm_title
                      , cm_file_name):
        """Provide a full picture of the model performance and accuracy

        :param model: an initiated machine learning model
        :type model: Any
        :param param_dist: a dictionary with the parameters and their respective ranges for the tuning
        :type param_dist: dict
        :param cm_title: title for the confusion matrix plot
        :type cm_title: str
        :param cm_file_name: title for the confusion matrix plot
        :type cm_file_name: str
        :return: the best model with its accuracy metrics and misclassified analysis
        :rtype: list
        """
        try:
            # Use random search to find the best hyperparameters
            rand_search = self.best_model(model=model, param_dist=param_dist)

            # transform the X_train and then get only the correct entries
            self.X_train = self.data_transformation_df(self.X_train_index, self.y_train)
            # subset the index for inspection
            self.X_train_fail = self.X_train[self.X_train['col0'].isnull()]
            #subset the data for the training
            self.X_train = self.X_train[~self.X_train['col0'].isnull()]

            self.X_test = self.data_transformation_df(self.X_test_index, self.y_test)
            self.X_test_fail = self.X_test[self.X_test['col0'].isnull()]
            # subset the data for the training
            self.X_test = self.X_test[~self.X_test['col0'].isnull()]

            #subset the y variables
            self.y_train = self.y_train.loc[self.X_train['train_index'].astype(int)]
            self.y_test = self.y_test.loc[self.X_test['train_index'].astype(int)]
            #here we need to make sure we subset the y variable for the correct index and remove the first two columns from X

            # fit the best model
            rand_search.fit(self.X_train[[x for x in self.X_train.columns if x not in ['train_index', 'file_index'] ]],
                            np.array(self.y_train).ravel())

            # generate predictions with the best model
            y_pred = rand_search.predict(self.X_test[[x for x in self.X_test.columns if x not in ['train_index', 'file_index'] ]])

            # calculate model accuracy
            acc, precision, recall = self.accuracy_metrics(y_pred=y_pred,
                                                           cm_title=cm_title,
                                                           cm_file_name=cm_file_name)

            # check the misclassified datapoints
            misclassified = self.misclassified_analysis(y_pred=y_pred)

            logging.info('Model Results Calculated')
        except Exception as error:
            logging.error(error)

        return acc, precision, recall,  rand_search , misclassified

    def random_forest_results(self):
        """Run Random Forest and conduct hyperparameter tuning, accuracy measurement and feature importance

        :return: accuracy, precision, recall, confusion matrix plot file with the name 'rf_confusion_matrix.png', misclassified analysis, feature importance plot with the name 'rf_feature_importance.png'
        :rtype: list
        """
        try:
            param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(1, 20)}
            # Create a random forest classifier
            rf = RandomForestClassifier()

            # check the model results
            acc, precision, recall,  rand_search,misclassified = self.model_results(model=rf, param_dist=param_dist,
                                                                                    cm_title='RF Confusion Matrix',
                                                                                    cm_file_name='rf_confusion_matrix.png')

            # # check the features importance
            best_model = rand_search.best_estimator_
            importances = best_model.feature_importances_
            self.forest_importances = pd.Series(importances, index=[x for x in self.X_train.columns if x not in ['train_index', 'file_index']])

            code_str = """
            self.forest_importances.sort_values(ascending=False).plot(kind='barh')
            plt.ylabel('Importance')
            plt.xlabel('Features')
                        """

            self.plot_figure(
                plot_title='RF Feature Importance'
                , file_title='rf_feature_importance.png'
                , plot_code=code_str
            )
            logging.info('Random forest results calculated')

            # create a plot for the misclassified
            self.misclass_rf = pd.DataFrame(misclassified)
            self.misclass_rf.reset_index(inplace=True)
            #self.misclass_rf.sort_values(ascending=False, by=self.old_col_name, inplace=True)

            code_str = "sns.barplot(self.misclass_rf, x=self.old_col_name, y='count')"

            self.plot_figure(
                plot_title='Random Forest Misclassified Distribution'
                , file_title='misclassified_rf.png'
                , plot_code=code_str
            )
            logging.info("Random Forest misclassified distribution plot created")

            return acc, precision, recall , misclassified
        except Exception as error:
            logging.error(error)
