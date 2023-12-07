# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

class BeeNotBee:
    """BeeNotBee class ....
    # TODO: update the documentation
    :param annotation_path: path to the annotation data
    :type annotation_path: str
    :param y_col: column label for the dependent variable in the annotation file
    :type y_col: str
    :param x_col: column label for the independent variable in the annotation file
    :type x_col: str




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
                 ,logname='bee.log'):
        self.annotation_path =annotation_path
        self.annotation_dtypes_path = annotation_dtypes_path
        self.x_col = x_col
        self.y_col = y_col
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
            logging.warning("NOT all data is in the correct type")


    def split_annotation_data(self,perc=0.25):
        """
        Split the annotation data into train and test. Save the csv files
        :param perc: test split percentage, a number between 0.0 and 1.0
        :type perc: float
        :return: X train, X test, y train and y test pandas data frames
        :rtype: pandas.DataFrame
        """
        if isinstance(perc, float) and perc >= 0.0 and perc <= 1.0:

        else:
            print('no')
        self.X_train_index, self.X_test_index, self.y_train, self.y_test = train_test_split(self.annotation_df[[self.x_col]],
                                                                        self.annotation_df[[self.y_col]],
                                                                        test_size=perc)

        # save all files for reproducibility
        self.X_train_index.to_csv('X_train_index.csv')
        self.X_test_index.to_csv('X_test_index.csv')
        self.y_train.to_csv('y_train.csv')
        self.y_test.to_csv('y_test.csv')






