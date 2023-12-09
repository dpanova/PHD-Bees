# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
import numpy as np
from scipy.fft import fft,fftfreq

class BeeNotBee:
    """BeeNotBee class ....
    # TODO: update the documentation
    :param annotation_path: path to the annotation data
    :type annotation_path: str
    :param y_col: column label for the dependent variable in the annotation file
    :type y_col: str
    :param x_col: column label for the independent variable in the annotation file
    :type x_col: str
    :param bee_folder: folder name where only bee acoustic files are hosted
    :type bee_folder: str
    :param no_bee_folder: folder name where only no-bee acoustic files are hosted
    :type no_bee_folder: str



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
                 ,logname='bee.log'
                 ,bee_folder = 'data/bee/'
                 ,no_bee_folder = 'data/nobee/'):
        self.annotation_path =annotation_path
        self.annotation_dtypes_path = annotation_dtypes_path
        self.x_col = x_col
        self.y_col = y_col
        self.bee_folder = bee_folder
        self.no_bee_folder = no_bee_folder
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)


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


    def split_annotation_data(self,perc=0.25):
        """
        Split the annotation data into train and test. Save the csv files
        :param perc: test split percentage, a number between 0.0 and 1.0
        :type perc: float
        :return: X train, X test, y train and y test pandas data frames
        :rtype: pandas.DataFrame
        """
        if isinstance(perc, float) and perc >= 0.0 and perc <= 1.0:
            logging.info('Percentage number is NOT correct')
            if (self.x_col in self.annotation_df) and (self.x_col in self.annotation_df):
                logging.info('X_col and y_col are valid columns')
                self.X_train_index, self.X_test_index, self.y_train, self.y_test = train_test_split(self.annotation_df[[self.x_col]],
                                                                                self.annotation_df[[self.y_col]],
                                                                                test_size=perc)

                # save all files for reproducibility
                self.X_train_index.to_csv('X_train_index.csv')
                self.X_test_index.to_csv('X_test_index.csv')
                self.y_train.to_csv('y_train.csv')
                self.y_test.to_csv('y_test.csv')
            else:
                logging.error('X_col and/or y_col are NOT valid columns')
        else:
            logging.error('Percentage number is NOT correct')

    def acoustic_file_names(self):
        """
        TODO add logging and checks
        Create a list of files which have bee and no-bee data
        :return: two lists - bee_files and nobee_files
        :rtype: list
        """
        self.bee_files = os.listdir(self.bee_folder)
        self.nobee_files = os.listdir(self.no_bee_folder)


    def harley_transformation_with_window(self,x,window = np.hanning):
        """
        TODO add logging and checks
        Transform np.array to fast harley transformation with a window function
        :param x: time sequence
        :type x: numpy array
        :param window: windowing function
        :type window: numpy function
        :return:
        """
        x_trans = fft(x * window(len(x)))  # adding window function
        x_trans = np.real(x_trans) - np.imag(x_trans)
        return x_trans

    def freq_powerspect_func(self,x, dt, npnts, denoise=True, cutoff=0):
        """
        TODO add loggings and checks
        Transform the FFT harley vector into frequency and power vector
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
        X = self.harley_transformation_with_window(x)
        n = np.arange(npnts)
        T = npnts * dt
        freq = n / T
        powerspect = 2 * np.abs(X) / npnts
        if denoise:
            powerspect = powerspect * (powerspect > cutoff)  # Zero all frequencies with small power
            X = X * (powerspect > cutoff)  # Zero small Fourier coefficients
        return freq, powerspect

    def binning(self, x, dt, npnts, n_bins =128, n_start = 0.0 , n_end = 2000.0, denoise=True, cutoff=0):
        """
        Transform a time to frequency using the freq_powerspect_func and then binning it to a specific number of frequency vectors. The power in every bin is the average of the original frequency vector.
        TODO add logging and checks
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
        return binned_x




