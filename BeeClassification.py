# libraries
import pandas as pd
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
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Bee:
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
            if (self.x_col in self.annotation_df) and (self.y_col in self.annotation_df):
                logging.info('X_col and y_col are valid columns')

                # here we need to update the documentation if everything works properly for the data split
                existing_indices = pd.DataFrame()
                existing_indices['index'] = [int(f.split('index')[1].split('.wav')[0]) for f in
                                             self.nobee_files + self.bee_files]
                existing_indices['Dir Exist'] = True

                self.annotation_df = self.annotation_df.merge(existing_indices, how='left', left_on='index', right_on='index')
                self.annotation_df['Dir Exist'].fillna(False, inplace=True)
                # 367 files are missing
                #here we may need to update something in future so that it is more universal
                annotation_df_updated  = self.annotation_df[self.annotation_df['Dir Exist']]
                annotation_df_updated = annotation_df_updated[annotation_df_updated['duration']>2.0]

                self.X_train_index, self.X_test_index, self.y_train, self.y_test = train_test_split(annotation_df_updated[[self.x_col]],
                                                                                annotation_df_updated[[self.y_col]],
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
        Create a list of files which have bee and no-bee data
        :return: two lists - bee_files and nobee_files
        :rtype: list
        """
        self.bee_files = os.listdir(self.bee_folder)
        self.nobee_files = os.listdir(self.no_bee_folder)
        logging.info('Files in the two directories are stores in two lists: bee_files and nobee_files')
        if len(self.bee_files)==0:
            raise ValueError('bee_files list is empty. Please, check the %s folder' %self.bee_folder)
        if len(self.nobee_files) == 0:
            raise ValueError('nobee_files list is empty. Please, check the %s folder' % self.no_bee_folder)

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
    def data_transformation_row(self, arg):
        """
        A row-wise function which finds the correct file from the annotation data frame and then transforms the acoustic data to binned harley fft vector. Stores the index from the annotation data frame (the key) and the df index to track the associated y values.
        :param arg: tuple with first argument the index of each row of a data frame, second argument - the actual row of the data frame and third argument - data frame with the dependant variable
        :type arg: tuple

        :return: list of lists with the transformed data. The non-existent files are returned as None type.
        :rtype: list
        """
        if type(arg) != tuple:
            raise ValueError('Invalid arg type. arg is type %s and expected type is tuple.' %type(arg).__name__)
        train_index, row,y = arg

        # get the necessary indices to trace files easily
        file_index = row['index']  # this index is necessary to ensure we have the correct file name (coming from the annotation file)
        label = y.loc[train_index, 'label']
        # check if the file from the annotation data exists in the folders
        try:
            if label == 'bee':
                file_name = [x for x in self.bee_files if x.find('index' + str(file_index) + '.wav') != -1][0]
                file_name = self.bee_folder + file_name
            else:
                file_name = [x for x in self.nobee_files if x.find('index' + str(file_index) + '.wav') != -1][0]
                file_name = self.no_bee_folder + file_name
            logging.info('%s file exists.' %file_name)
        except:
            logging.warning('No file with index %s exists.' %str(file_index))

        # read the files
        try:
            samples, sample_rate = librosa.load(file_name, sr=None, mono=True, offset=0.0, duration=None)
            # check if the data extraction is correct
            duration_of_sound =  round(len(samples) / sample_rate,2)
            annotation_duration = self.annotation_df.loc[self.annotation_df['index'] == file_index, 'duration']

            # we need to do different error handling with log files at a later point
            if duration_of_sound == round(annotation_duration.loc[train_index,],2):
                logging.info('%s file has the correct duration.' % file_name)
            else:
                logging.warning('%s file DOES NOT have the correct duration.' % file_name)

            # transform the time to binned frequency vector
            dt = 1 / sample_rate
            t = np.arange(0, duration_of_sound, dt)
            npnts = len(t)
            sample_transformed = self.binning(x=samples, dt=dt, npnts=npnts)

            # we need to add the indices for tracking purposes
            sample_transformed.insert(0, file_index)
            sample_transformed.insert(0,train_index)


            logging.info('%s file transformed and added to the transformed data frame' % file_name)
            return sample_transformed
        except:
            return [train_index,file_index]
            logging.warning('File with index %s is NOT added to the transformed data frame' %str(file_index))


    def data_transformation_df(self, X,y):
        """
        Find the correct file from the annotation data frame and then transform the acoustic data to binned harley fft vector. Store the index from the annotation data frame (the key) and the df index to track the associated y values.
        :param X: pandas data frame with the indices of the acoustic files which need to be transformed
        :type X: pandas.DataFrame
        :param y: pandas data frame with the dependant variable
        :type y: pandas.DataFrame

        :return: data frame with the transformed data
        :rtype:pandas.dataFrame
        """
        if type(X) != pd.core.frame.DataFrame:
            raise ValueError('Invalid arg type. arg is type %s and expected type is pandas.core.frame.DataFrame.' % type(X).__name__)
        if type(y) != pd.core.frame.DataFrame:
            raise ValueError('Invalid arg type. arg is type %s and expected type is pandas.core.frame.DataFrame.' % type(y).__name__)
        if 'index' not in X.columns.to_list():
            raise ValueError('Column index is not part of X data frame. It is a requirement.')
        if 'label' not in y.columns.to_list():
            raise ValueError('Column label is not part of y data frame. It is a requirement.')
        pool = mp.Pool(processes=mp.cpu_count())
        X_transformed = pool.map(self.data_transformation_row,[(train_index, row,y) for train_index, row in X.iterrows()])
        # add the column names
        cols = ['train_index','file_index']
        max_length = max([len(x) for x in X_transformed if x is not None])
        cols = cols + ['col' + str(x) for x in range(max_length - 2)]
        # transform to data frame
        X_df = pd.DataFrame(columns=cols)
        for x in X_transformed:
            if len(x) !=2:
                X_df.loc[len(X_df)] = x
            else:
                x_updated = x+[None]*(max_length-len(x))
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
