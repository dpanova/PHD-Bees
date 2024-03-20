import logging
from random import randint
from time import sleep
from selenium.webdriver.common.keys import Keys
import undetected_chromedriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from auxilary_functions import recommendations, reads, citations, cos_func
import nltk
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx
import pandas as pd
import time
import itertools
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class BeeLitReview:
    """
    TODO update the description and the validations
    """
    def __init__(self
                 ,logname='BeeLitReview.log'
                 ,already_scraped=True
                 ,scraped_file_name = 'scraped_articles.csv'
                 ,scraped_file_validation = 'scraped_articles_validation.csv'):
        self.scraped_file_name = scraped_file_name
        self.similarity_df = pd.DataFrame()
        self.embeddings = None
        self.graph_start = None
        self.graph_end = None
        self.scraped_file_validation = scraped_file_validation
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)
        if already_scraped:
            self.df = pd.read_csv(scraped_file_name)
            self.validate_scraped_csv()
        else:
            self.df = pd.DataFrame()

        if type(already_scraped) != bool:
            raise ValueError(
                'Invalid already_scraped type. It is type %s and expected type is str.' % type(already_scraped).__name__)
        if type(scraped_file_name) != str:
            raise ValueError(
                'Invalid scraped_file_name type. It is type %s and expected type is str.' % type(scraped_file_name).__name__)
        if not scraped_file_name.endswith('csv'):
            raise ValueError(
                '%s input is not the correct type. It should be .csv extension' % scraped_file_name)
        if type(scraped_file_validation) != str:
            raise ValueError(
                'Invalid scraped_file_validation type. It is type %s and expected type is str.' % type(scraped_file_validation).__name__)
        if not scraped_file_validation.endswith('csv'):
            raise ValueError(
                '%s input is not the correct type. It should be .csv extension' % scraped_file_validation)
    def validate_scraped_csv(self):
        """"
        Validate scraped data
        :return: error if the data is not as expected
        """
        dtypes_df = pd.read_csv(self.scraped_file_validation)
        cols = dtypes_df['col_name']
        if len(set(cols).difference(set(self.df))) == 0:
            logging.info("All columns are present correctly")
        else:
            raise ValueError(
                'NOT all required columns are present')
        # check that the data types are as expected
        data_type = pd.DataFrame(self.df.dtypes)
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
            raise ValueError(
                'NOT all data is in the correct type')
            

    def article_scrape(self, article):
        """
        Function to scrape the important aspects from an article result - title, category, date, reads, citations, authors, url and abstract
        :param article: article result
        :type article: undetected_chromedriver.webelement.WebElement
        :return: scraped date in the df object
        :rtype: dataframe
        """
        if type(article) != undetected_chromedriver.webelement.WebElement:
            raise ValueError(
                'Invalid article type. It is type %s and expected type is undetected_chromedriver.webelement.WebElement.' % type(article).__name__)

        # access the inner item stack for each result
        items = article.find_elements(By.CLASS_NAME, 'nova-legacy-v-entity-item__stack-item')
        # get the title from the first tile
        title = items[0].find_element(By.XPATH, 'div//a').text
        logging.info(title)
        # get the url
        url = items[0].find_element(By.XPATH, 'div//a').get_attribute('href')

        # get the left categories
        temp_left = items[1].find_elements(By.CLASS_NAME, 'nova-legacy-v-entity-item__meta-left')

        # since it may have more than one left category, we will iterate over it
        if len(temp_left) > 1:
            left_list = []
            for i in range(len(temp_left)):
                left_list.append(temp_left[i].text)
            left = ';'.join(left_list)
        else:
            left = temp_left[0].text

        # get the date
        temp_right = items[1].find_elements(By.CLASS_NAME, 'nova-legacy-v-entity-item__meta-right')
        date = temp_right[0].find_elements(By.XPATH, 'ul//li')[0].text

        # get the right category
        cat = temp_right[0].find_elements(By.XPATH, 'ul//li')
        if len(cat) == 1:
            right = ''
        else:
            right_list = []
        for i in range(1, len(cat)):
            right_list.append(cat[i].text)
        right = ';'.join(right_list)

        # get the authors

        authors_list = items[3].find_elements(By.XPATH, 'ul//li//a//span')
        if len(authors_list) > 1:
            temp_authors_list = []
            for author in authors_list:
                if len(author.text) > 0:
                    temp_authors_list.append(author.text)
            author = ';'.join(temp_authors_list)
        else:
            author = authors_list[0].text

        # get the abstract
        abstract = items[4].text

        # append the results to the DF
        page_dict = {'Title': [title],
                     'URL': [url],
                     'Authors': [author],
                     'Date': [date],
                     'Left Category': [left],
                     'Right Category': [right],
                     'Abstract': [abstract]
                     }
        page_df = pd.DataFrame(page_dict)
        self.df = pd.concat([self.df, page_df], ignore_index=True)
    def researchgate_scraper(self
                , username
                ,password
                ,url='https://www.researchgate.net/login?_sg=ITbsht6C-ko8ZSF49e9deV1BNgFqKApIltdEguj_4uiZ9K_WzI6gYtnTL5xsgogphkn5Z2RJTNuYqd9fMiGJfg'
                , query='bee acoustic'
                 ):
        """
        Function to scrape the articles from ResearchGate
        :param username: username in the website
        :type username: str
        :param password: password in the website
        :type password: str
        :param url: url of the website
        :type url: str
        :param query: query to search in the website
        :type query: str
        :return: scraped date in the df object and saved csv
        :rtype: dataframe
        """

        if type(username) != str:
            raise ValueError(
                'Invalid username type. It is type %s and expected type is str.' % type(username).__name__)

        if type(password) != str:
            raise ValueError(
                'Invalid password type. It is type %s and expected type is str.' % type(password).__name__)
        if type(url) != str:
            raise ValueError(
                'Invalid url type. It is type %s and expected type is str.' % type(url).__name__)
        if type(query) != str:
            raise ValueError(
                'Invalid query type. It is type %s and expected type is str.' % type(query).__name__)

        driver = uc.Chrome()
        driver.get(url)
        #add password and username
        uname = driver.find_element("id", "input-login")
        uname.send_keys(username)
        pword = driver.find_element("id", "input-password")
        pword.send_keys(password)
        #click submit
        driver.find_element(By.XPATH, '//button[@type="submit"]').click()
        # search for the specific query
        search = driver.find_element('name', 'query')
        search.send_keys(query)
        search.send_keys(Keys.ENTER)
        #get only those with full text available
        driver.find_element(By.CLASS_NAME, 'fulltext-availability-switch').click()
        #create the data frame which will hold the results
        self.df = pd.DataFrame(
            columns=['Title', 'URL', 'Authors', 'Date', 'Left Category', 'Right Category', 'Abstract'])

        # get the results for the first page
        result_items = driver.find_elements(By.CLASS_NAME, 'search-box__result-item')

        for article in result_items:
            self.article_scrape(article)

        #save the data
        self.df.to_csv(self.scraped_file_name, index=False)

        #get the results for the rest of the pages
        current_url = driver.current_url
        for step in range(0, 1000, 10):
            main_url = current_url + '&limit=10&offset=' + str(step)
            driver.get(main_url)
            logging.info(main_url)
            sleep(randint(10, 30))

            # do the same steps
            result_items = driver.find_elements(By.CLASS_NAME, 'search-box__result-item')
            for article in result_items:
                self.article_scrape(article)

            # save the data
            self.df.to_csv(self.scraped_file_name, index=False)
        logging.info('Data scraped successfully!')

    def scraped_data_enhancement(self
                                 ,stats_col = 'Right Category'
                                 ,cat_col ='Left Category'
                                 ,key_col ='Title'
                                 ,abstract_col ='Abstract'
                                 ,date_col='Date'):
        """
        Function to extract dates, result type - article/ etc, tokenize the abstract and split the important stats such as citations.
        :param stats_col: Column which has the stats data
        :type stats_col: str
        :param cat_col:Column which has the result type
        :type cat_col: str
        :param key_col: Column which will be used for the duplication reduction (last entry is kept in the data)
        :type key_col: str
        :param abstract_col: Column which has the abstract of the text
        :type abstract_col: str
        :param date_col: Colum which has the date information
        :type date_col: str
        :return: Returns pandas with new columns with the extracted information, refer to the df in the object. Additionally, the data is saved locally to _enhanced file
        """
        if type(stats_col) != str:
            raise ValueError(
                'Invalid stats_col type. It is type %s and expected type is str.' % type(stats_col).__name__)
        if type(cat_col) != str:
            raise ValueError(
                'Invalid cat_col type. It is type %s and expected type is str.' % type(cat_col).__name__)
        if type(key_col) != str:
            raise ValueError(
                'Invalid key_col type. It is type %s and expected type is str.' % type(key_col).__name__)
        if type(abstract_col) != str:
            raise ValueError(
                'Invalid abstract_col type. It is type %s and expected type is str.' % type(abstract_col).__name__)
        if type(date_col) != str:
            raise ValueError(
                'Invalid date_col type. It is type %s and expected type is str.' % type(date_col).__name__)

        #ensure we don't have duplicates
        self.df.drop_duplicates(subset=[key_col], keep='last', inplace=True)
        self.df['Citations'] = self.df[stats_col].apply(lambda x: citations(x))
        logging.info('Citations extracted successfully')
        self.df['Recommendations'] = self.df[stats_col].apply(lambda x: recommendations(x))
        logging.info('Recommendations extracted successfully')
        self.df['Reads'] = self.df[stats_col].apply(lambda x: reads(x))
        logging.info('Reads extracted successfully')

 
        # extracting type of text from the left category
        self.df['Text Type'] = self.df[cat_col].apply(
            lambda x: [a for a in x.split(';') if a != 'Full-text available'][0])
        self.df['Text Type'].value_counts()  # for teh report
        logging.info('Text type extracted successfully')

        self.df.reset_index(inplace=True)
        self.df['Abstract Count Words'] = self.df[abstract_col].apply(lambda x: len(nltk.word_tokenize(x)))
        logging.info('Abstract tokenized successfully')
        self.df['Datetime'] = self.df[date_col].apply(lambda x: datetime.strptime(x, '%B %Y'))
        self.df['Year'] = self.df['Datetime'].apply(lambda x: x.year)
        logging.info('Date extracted successfully')

        #save the new data for tracking purposes
        self.df.to_csv(self.scraped_file_name.split('.csv')[0]+'_enhanced.csv',index=False)
        logging.info('Data saved successfully to %s' %self.scraped_file_name.split('.csv')[0]+'_enhanced.csv')

    def encode_with_transformers(self
                                 ,model_id='sentence-transformers/all-mpnet-base-v2'
                                 ,abstract_col='Abstract'):
        #TODO add return as function and add description and checks
        """
        Function to encode the abstract text with HuggingFace transformers. Embeddings are saved to embeddings
        :param model_id: HuggingFace transformers' model
        :type model_id: str
        :param abstract_col: Column indicating the abstract
        :type abstract_col: str
        :return: embeddings in a list format
        """
        if type(model_id) != str:
            raise ValueError(
                'Invalid model_id type. It is type %s and expected type is str.' % type(model_id).__name__)
        if type(abstract_col) != str:
            raise ValueError(
                'Invalid abstract_col type. It is type %s and expected type is str.' % type(abstract_col).__name__)
        #initiate the model
        model = SentenceTransformer(model_id)
        abstract_list = list(self.df[abstract_col])
        embeddings = model.encode(abstract_list)
        self.embeddings = embeddings
        logging.info('Embeddings created for the abstracts.')


    def calculate_similarity(self):
        """
        Calculate the pairwise cosine similarity between each abstract based on the already calculated embeddings
        :return: similarity dataframe with columns pair0, pair1 and cos. Saves the data to similarity_df.csv
        """
        embeddings_index = range(len(self.embeddings))
        pairs = list(itertools.combinations(embeddings_index, 2))
        similarity_df = pd.DataFrame()
        for pair in pairs:
            print(pair)
            a0 = np.array(self.embeddings[pair[0]])
            a1 = np.array(self.embeddings[pair[1]])
            cos_sim = cos_func(a0, a1)
            temp_dict = {'pair0': pair[0], 'pair1': pair[1], 'cos': cos_sim}
            temp_df = pd.DataFrame([temp_dict])
            similarity_df = pd.concat([similarity_df, temp_df], ignore_index=True)
        self.similarity_df = similarity_df
        #save the similarity data locally
        similarity_df.to_csv('similarity_df.csv', index=False)
        logging.info('Pairwise cosine similarity calculated.')

    def calculate_TSP(self
                      ,num_articles = 50
                      ,path_file_name = 'path.csv'
                      ,to_review_file_name = 'to_review.csv'):
        """
        Calculates the Travel Salesmen Path based on the similarity between the abstracts.
        :param num_articles: number of articles to bee looked into
        :type num_articles: int
        :param path_file_name: name of the file to store the path
        :type path_file_name: str
        :param to_review_file_name: name of the file to store the review file
        :type to_review_file_name: str
        :return: Returns dataframe with the first num_articles and saves to data to to_review_file_name.csv for further inspection
        """
        if type(num_articles) != int:
            raise ValueError(
                'Invalid num_articles type. It is type %s and expected type is int.' % type(num_articles).__name__)
        if type(path_file_name) != str:
            raise ValueError(
                'Invalid path_file_name type. It is type %s and expected type is str.' % type(path_file_name).__name__)
        if not path_file_name.endswith('csv'):
            raise ValueError(
                '%s input is not the correct type. It should be .csv extension' % path_file_name)
        if type(to_review_file_name) != str:
            raise ValueError(
                'Invalid to_review_file_name type. It is type %s and expected type is str.' % type(to_review_file_name).__name__)
        if not to_review_file_name.endswith('csv'):
            raise ValueError(
                '%s input is not the correct type. It should be .csv extension' % to_review_file_name)
        G = nx.from_pandas_edgelist(self.similarity_df, source='pair0', target='pair1', edge_attr='cos')
        tsp = nx.approximation.traveling_salesman_problem
        self.graph_start = time.time()
        path = tsp(G, cycle=False)
        self.graph_end = time.time()
        logging.info('Graph created and TSP calculated.')

        pd.DataFrame(path).to_csv(path_file_name, index=False)
        # get the first 50th
        to_review_df = self.df.loc[path[:num_articles], ['index', 'Title', 'Abstract']]
        to_review_df['Choose'] = False
        to_review_df.to_csv(to_review_file_name, index=False)
        logging.info('%s file saved.' % to_review_file_name)


    #TODO create a pdf report for both graph and clusters
    # review.df['Year'].value_counts()  # for the report
    # review.df['Abstract Count Words'].hist()  # for teh report
    # sum(review.df['Abstract Count Words'] <= 384.0) / len(review.df)  # 92% for the report
    # review.df['Text Type'].value_counts()  # for teh report
    # similarity_df['cos'].hist()
    # print("The time of execution of above program is :",
    #       (end - start) * 10 ** 3, "ms")
