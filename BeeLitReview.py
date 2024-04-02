import logging
from random import randint
from time import sleep
from selenium.webdriver.common.keys import Keys
# import undetected_chromedriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from functools import partial
import multiprocessing as mp
from auxilary_functions import recommendations, reads, citations, include_tuple, cos_sim_func, start_page, h0, h1, h2, normal_text, pdf_graph, pdf_table


from sentence_transformers import SentenceTransformer
import numpy as np
import networkx as nx

import time
import itertools
from clusteval import clusteval

from auxilary_functions import pd_to_tuple
from fpdf import FPDF
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.snowball import SnowballStemmer
import nltk



from langdetect import detect
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class BeeLitReview:
    """
    TODO update the description and the validations
    """
    def __init__(self
                 ,logname='BeeLitReview.log'
                 ,already_scraped=True
                 ,query='bee acoustic machine learning'
                 ,scraped_file_name = 'scraped_articles.csv'
                 ,scraped_file_validation = 'scraped_articles_validation.csv'
                 ,to_review_file_name='to_review.xlsx'):
        self.scraped_file_name = scraped_file_name
        self.query = query
        self.similarity_df = None
        self.embeddings = None
        self.graph_start = None
        self.graph_end = None
        self.df_raw = None
        self.path = None
        self.most_similar = None
        self.cosine_clustering_threshold = None
        self.year = None
        self.type = None
        self.read = None
        self.citation = None
        self.to_review_df = None
        self.clustering_indices = None
        self.to_cluster_embeddings = None
        self.ce_hdbscan_results = None
        self.ce_agg_results = None
        self.to_review_file_name = to_review_file_name
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
        """
        :param query: query to search in the website
        :type query: str
        """
        if type(query) != str:
            raise ValueError(
                'Invalid query type. It is type %s and expected type is str.' % type(query).__name__)
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
        if type(to_review_file_name) != str:
            raise ValueError(
                'Invalid to_review_file_name type. It is type %s and expected type is str.' % type(to_review_file_name).__name__)
        if not to_review_file_name.endswith('xlsx'):
            raise ValueError(
                '%s input is not the correct type. It should be .xlsx extension' % to_review_file_name)
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
        if type(article) != uc.webelement.WebElement:
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
        try:
            authors_list = items[3].find_elements(By.XPATH, 'ul//li//a//span')
            if len(authors_list) > 1:
                temp_authors_list = []
                for author in authors_list:
                    if len(author.text) > 0:
                        temp_authors_list.append(author.text)
                author = ';'.join(temp_authors_list)
            else:
                author = authors_list[0].text
        except:
            author ='No author'
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
                , username = ''
                ,password = ''
                ,url='https://www.researchgate.net/login?_sg=ITbsht6C-ko8ZSF49e9deV1BNgFqKApIltdEguj_4uiZ9K_WzI6gYtnTL5xsgogphkn5Z2RJTNuYqd9fMiGJfg'

                 ):
        """
        Function to scrape the articles from ResearchGate
        :param username: username in the website
        :type username: str
        :param password: password in the website
        :type password: str
        :param url: url of the website
        :type url: str
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



        driver = uc.Chrome()
        driver.get(url)
        sleep(3)
        # click the accept button
        driver.find_element(By.CLASS_NAME, 'didomi-regular-notice').find_element(By.CLASS_NAME,
                                                                                 'didomi-buttons').find_elements(
            By.CLASS_NAME, "didomi-components-button")[2].click()
        sleep(5)
        #add password and username
        uname = driver.find_element("id", "input-login")
        sleep(3)
        uname.send_keys(username)

        pword = driver.find_element("id", "input-password")
        sleep(3)
        pword.send_keys(password)
        #click submit
        driver.find_element(By.XPATH, '//button[@type="submit"]').click()
        # search for the specific query
        sleep(3)
        search = driver.find_element('name', 'query')
        search.send_keys(self.query)
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
            sleep(randint(10, 45))

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
        Function to extract dates, result type - article/ etc, tokenize the abstract, split the important stats such as citations and keep only the English text.
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
        #create a new data frame to host the raw data
        self.df_raw = self.df
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

        #check what is the language of the abstract
        self.df_raw['Language'] = self.df_raw[abstract_col].apply(lambda x: detect(x))
        #make sure that only English language is left in the enhanced version
        self.df = self.df[self.df[abstract_col].apply(lambda x: detect(x))=='en']
        logging.info('Only English abstracts are left')

        #reset the index for trackability
        self.df.reset_index(inplace=True)

        #save the new data for tracking purposes
        self.df.to_csv(self.scraped_file_name.split('.csv')[0]+'_enhanced.csv',index=False)
        logging.info('Data saved successfully to %s' %self.scraped_file_name.split('.csv')[0]+'_enhanced.csv')

    def encode_with_transformers(self
                                 ,model_id='sentence-transformers/all-mpnet-base-v2'
                                 ,abstract_col='Abstract'):
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
        #add the query embedding as the last one
        abstract_list.append(self.query)
        embeddings = model.encode(abstract_list)
        self.embeddings = embeddings
        logging.info('Embeddings created for the abstracts.')


    def calculate_similarity(self):
        """
        Calculate the pairwise cosine similarity between each abstract based on the already calculated embeddings
        :return: similarity dataframe with columns pair0, pair1 and cos. Saves the data to similarity_df.csv
        """
        #initiate the multiprocessing
        pool = mp.Pool(processes=mp.cpu_count())

        query_index = len(self.embeddings)
        embeddings_index = range(len(self.embeddings) - 1)

        #calculate the similarity of each pair, excluding the query
        pairs = list(itertools.combinations(embeddings_index, 2))
        similarity = pool.map(partial(cos_sim_func, embedding_list=self.embeddings), pairs)
        similarity_df = pd.DataFrame(similarity)

        #calculate the similarity of each abstract and the query
        pairs = [((query_index - 1), x) for x in embeddings_index]
        most_similar = pool.map(partial(cos_sim_func, embedding_list=self.embeddings), pairs)
        most_similar_df = pd.DataFrame(most_similar)
        #identify the most similar one and add it with cosine zero, this will be our initial point
        most_similar_df = most_similar_df[most_similar_df['cos'] == most_similar_df['cos'].max()]
        most_similar_df['cos'] = 0
        self.most_similar = most_similar_df['pair1']

        # add it to the similarity df
        self.similarity_df = pd.concat([similarity_df, most_similar_df], ignore_index=True)
        #save the similarity data locally
        similarity_df.to_csv('similarity_df.csv', index=False)
        logging.info('Pairwise cosine similarity calculated.')

    def calculate_TSP(self
                      ,num_articles = 50
                      ,path_file_name = 'path.csv'):
        """
        Calculates the Travel Salesmen Path based on the similarity between the abstracts.
        :param num_articles: number of articles to bee looked into
        :type num_articles: int
        :param path_file_name: name of the file to store the path
        :type path_file_name: str
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

        G = nx.from_pandas_edgelist(self.similarity_df, source='pair0', target='pair1', edge_attr='cos')
        tsp = nx.approximation.traveling_salesman_problem
        self.graph_start = time.time()
        path = tsp(G, cycle=False)
        self.graph_end = time.time()
        logging.info('Graph created and TSP calculated.')
        self.path = path
        ind = self.path.index(self.most_similar.iloc[0])
        if ind >= num_articles:
            left_length = len(self.path[ind:-1])
            left = self.path[ind:-1]
            right = self.path[:(num_articles-left_length)]
            to_review_path = left + right
        else:
            to_review_path = self.path[:num_articles]

        pd.DataFrame(path).to_csv(path_file_name, index=False)
        # get the first 50th
        to_review_df = self.df.loc[to_review_path, ['index', 'Title', 'Abstract']]
        to_review_df['Choose'] = False
        # self.to_review_df = to_review_df
        to_review_df.to_excel(self.to_review_file_name, index=False)
        logging.info('%s file saved.' % self.to_review_file_name)


    def clustering_evaluation(self
                        , cosine_threshold = 0.3
                        ,year=2023
                        ,type = ['Article', 'Conference Paper','Preprint','Patent','Thesis']
                        ,read = 10
                        ,citation = 1):

        """
        TODO update here 
        Create two types of clustering - HDBSCAN and agglomerative. Additionally, we can filter the abstracts with specific text type, reads and citations. Moreover we can choose those are highly correlated with the originally chosen abstracts.  
        :param cosine_threshold: threshold for cosine similarity with the chosen articles. The value is between 0 and 1 
        :type cosine_threshold: float
        :param year: year of the abstract to filter
        :type year: int
        :param type: type of the abstract to filter. You can choose from  %s
        :type type: list
        :param read: number of reads an abstract should be associated with 
        :type read: int
        :param citation: number of citations an abstract should be associated with
        :type citation: int
        :return: pandas data frame with the results (labeled abstracts) 
        :rtype: pandas data frame
        """ % ';'.join(list(self.df['Text Type'].unique()))
        self.cosine_clustering_threshold = cosine_threshold
        self.year = year
        self.type = type
        self.read = read
        self.citation = citation
        logging.info("Save cluster parameters for articles' filtering")

        self.to_review_df = pd.read_excel(self.to_review_file_name)
        self.to_review_df['Choose'] = self.to_review_df['Choose'].astype('bool')
        logging.info("Identify articles of interest")
        index_chosen = list( self.to_review_df[ self.to_review_df['Choose']]['index'])

        self.similarity_df['Under Inspection'] = False
        self.similarity_df['key'] = list(zip(self.similarity_df['pair0'], self.similarity_df['pair1']))

        for index, row in self.similarity_df.iterrows():
            self.similarity_df.loc[index, 'Under Inspection'] = include_tuple(index_chosen, row, t_limit=cosine_threshold)

        pair0 = list(self.similarity_df[self.similarity_df['Under Inspection']]['pair0'])
        pair1 = list(self.similarity_df[self.similarity_df['Under Inspection']]['pair1'])
        pair = pair0+pair1
        filter_index = list(self.df[(self.df['Year'] >= year) & (self.df['Text Type'].isin(type)) & (
                    self.df['Citations'] >= citation) & (self.df['Reads'] >= read)]['index'])
        common_index = list(set(pair).intersection(filter_index))

        #save the clustering indices in the object
        self.clustering_indices= list(set(index_chosen + common_index))
        self.to_cluster_embeddings = np.vstack([self.embeddings[x] for x in self.clustering_indices])
        logging.info("Extract the results of interest")
        #HDBSAN
        ce_hdbscan = clusteval(cluster='hdbscan', evaluate='silhouette', linkage='complete')
        self.ce_hdbscan_results = ce_hdbscan.fit(self.to_cluster_embeddings)
        ce_hdbscan.fit(self.to_cluster_embeddings)
        # ce_hdbscan.plot()[0].savefig("sil_score_hdbsan.png")

        ce_hdbscan.dendrogram()
        plt.savefig("dendogram_hdbscan.png")
        logging.info("HDBSCAN clustering complete")

        #Hierarchical
        ce_agg = clusteval(cluster='agglomerative', evaluate='silhouette', linkage='complete', metric='cosine')

        # Cluster evaluation
        self.ce_agg_results = ce_agg.fit(self.to_cluster_embeddings)
        #save plots for the report
        ce_agg.fit(self.to_cluster_embeddings)


        ce_agg.dendrogram()
        plt.savefig("dendogram_agg.png")
        logging.info("Agglomeration clustering complete. ")



    def pdf_report_generate(self, report_file_name='lit_report.pdf'):
        """
        Generate automated pdf report based on the results from the class
        The report has:
        - INTRO: Report title, author, data of the report and disclaimer and dependant variable distribution
        - DATA OVERVIEW - Time, word, result type, language distributions
        - Optimal ML Path
            - Travelling Salesman Problem Solution
            - Similarity Difference
            - Time Saved
            - TSP Results of Interest
            - Clustering Results
        - Suggested Reads
        :return: pdf file
        :rtype: pdf
        """
        if type(report_file_name) != str:
            raise ValueError(
                'Invalid report_file_name type. It is type %s and expected type is str.' % type(report_file_name).__name__)
        if not report_file_name.endswith('pdf'):
            raise ValueError(
                '%s input is not the correct type. It should be .pdf extension' % report_file_name)

        # Generate the PDF report and save it
        pdf = FPDF('P', 'mm', 'A4')
        logging.info("Create pdf.")
        # PAGE 1

        # INTRO
        start_page(pdf)

        h0('Literature self Automated Report', pdf)

        normal_text('Author: Denitsa Panova', pdf, x=3, italics=True)
        normal_text('Date: ' + datetime.today().strftime('%Y-%m-%d'), pdf, italics=True)

        text = (
            'Disclaimer: The objective of this report is to present the outcomes generated by automated literature research. '
            'A specialized interpretation is essential to derive accurate conclusions regarding correct articles to '
            'be selfed.')
        logging.info("Add intro.")
        normal_text(text, pdf, italics=True, x=10)

        h1('Data Overview', pdf)

        text = ('Query "%s" has been executed in researchgate.com and all available results are scraped - in total %s. '
                'Note, that in the context of this report, a result is a search result, it can be an article, '
                'presentation, etc.') % (self.query, str(len(self.df)))
        normal_text(text, pdf)
        logging.info("Add data overview")
        h2('Time Distribution', pdf)
        normal_text("Firstly, we will investigate how the results's distribution over the years.", pdf)

        # Table for the time distribution over years
        year_data = pd_to_tuple(self.df, 'Year')

        pdf_table(year_data, pdf, width=40, cols=(20, 20))
        logging.info("Add time distribution")
        h2('Word Distribution', pdf)

        text = (
                    "Then, we will investigate what is the average count of words per abstract. Here the goal is to see if the "
                    "default transformer model is still an adequate solution.The default model is all-mpnet-base-v2 and if the "
                    "word count is above 384, it truncates the text and we wouldn't have full results in the encoding stage. "
                    "We have %s of the results consenting the criteria."
                    % str(round(sum(self.df['Abstract Count Words'] <= 384.0) / len(self.df), 2)))

        normal_text(text, pdf)

        plot_code = "self.df['Abstract Count Words'].hist()"
        pdf_graph(pdf, plot_code=plot_code, x=50, y=206, w=110, h=0)
        logging.info("Add word distribution ")
        # NEW PAGE
        start_page(pdf)
        h2('Result type', pdf)
        text = (
            "Investigate what type of results are extracted. This will help in the correct choice for clustering the results. "
            "The default types for clustering are: Article and Conference Paper.")
        normal_text(text, pdf)
        type_data = pd_to_tuple(self.df, 'Text Type')
        pdf_table(type_data, pdf, width=60, cols=(40, 20))

        h2('Language', pdf)
        text = (
                    "Investigate what languages the results are in. We are considering only English languages for the purposes of "
                    "this research After removing non-English results, we lose %s of the data."
                    % str(round(len(self.df_raw[self.df_raw['Language'] != 'en']) / len(self.df_raw), 2)))

        normal_text(text, pdf)

        language_data = pd_to_tuple(self.df_raw, 'Language')
        pdf_table(language_data, pdf, width=40, cols=(40, 20))
        logging.info("Add language distribution ")
        h1('Optimal ML Path', pdf)
        h2('Traveling Salesmen Problem (TSP)', pdf)

        initial_point = str(self.most_similar.iloc[0])
        text = ("Each abstract is turned into an embedding, using the HuggingFace Transformer. The default one is "
                "all-mpnet-base-v2.After we calculate similarity between each scraped result with each another one, we apply TSP "
                "for finding the shortest path. The time for calculating it is %s ms. Additionally, to optimize the performance "
                "of the algorithm by creating an artificial connection in the graph between the query %s and the most similar "
                "result in terms of cosine similarity between embeddings. Then we artificially assigned similarity 0 to "
                "impose the algorithm to start from there. The result index is %s"
                % (str((self.graph_end - self.graph_start) * 10 ** 3), self.query, initial_point))

        normal_text(text, pdf)
        logging.info("Add TSP")
        h2('Similarity Difference', pdf)


        path = pd.DataFrame(self.path[:50])
        # create tuples based on the path pairs
        path_similarity = pd.DataFrame()
        path_similarity['pair0'] = list(path.iloc[:-1, 0])
        path_similarity['pair1'] = list(path.iloc[1:, 0])
        path_similarity = path_similarity.merge(self.similarity_df, how='inner', on=['pair0', 'pair1'])

        # let us do the same for the original research order
        original_similarity = pd.DataFrame()
        original_similarity['pair0'] = list(range(0, 49))
        original_similarity['pair1'] = list(range(1, 50))
        original_similarity = original_similarity.merge(self.similarity_df, how='inner', on=['pair0', 'pair1'])

        text = (
                    "In order to investigate what is the added value of TSP, we check what is the average similarity measure"
                    "in the first 50 TSP-suggested articles and the first 50 ResearchGate articles. TSP is %s and ResearchGate is %s."
                    % (str(round(path_similarity['cos'].mean(), 2)), str(round(original_similarity['cos'].mean(), 2))))

        normal_text(text, pdf)
        logging.info("Add similarity")
        h2('Time Saved', pdf)

        all_not_read_indix = list(set(list(self.df['index'])).difference(set(list(path.loc[:, 0]))))
        all_not_read_indix_max = [x for x in all_not_read_indix if x < max(list(list(path.loc[:, 0])))]
        # the cost in terms of time
        cost_time = str(
            round(self.df.loc[all_not_read_indix_max, 'Abstract Count Words'].sum() / 150 / 60, 2))  # 12 hours
        text = (
                    "Now we will calculate how much time it is saved by following the 50 articles suggested by TSP. The goal is to "
                    "understand how much time is saved and at the same time wider knowledge on the topic is acquired. The metric "
                    "encompasses the number of abstracts one should go over to read the same TSP-suggested results. According "
                    "to a research, one can read around 150 words per minute. Therefore, the saved time by going with the TSP "
                    "suggestion is %s hours." % cost_time)

        normal_text(text, pdf)
        logging.info("Add time saved")
        h2('TSP Results of Interest', pdf)
        normal_text('Let us look into the wordcloud of the TSP-results abstracts to see what those are in one look.',
                    pdf, x=0)


        words = nltk.word_tokenize(' . '.join(list(self.to_review_df['Title'])))

        stop_words = set(stopwords.words('english'))

        stemmer = SnowballStemmer("english")
        words = [stemmer.stem(word.lower())
                 for word in words
                 if word.isalnum() and word.lower() not in stop_words]
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

        plot_code = ("plt.imshow(wordcloud, interpolation='bilinear') \n"
                     "plt.axis('off')")
        pdf_graph(pdf, plot_code=plot_code, x=0, y=100, w=200, h=0)
        pdf.ln(115)
        normal_text(
            'The next step is to look into the TSP results and identify those which are of interest for the research.'
            'We have identified those:', pdf)

        logging.info("Add word cloud")
        for idx, row in self.to_review_df[self.to_review_df['Choose']].iterrows():
            normal_text('Original Index: ' + str(row['index']), pdf)
            normal_text('Title: ' + row['Title'], pdf)
        logging.info("Add results of interest")
        h2('Clustering Results', pdf)
        normal_text("Let us see the distribution of the HDBSCAN clusters.", pdf)

        hdbscan_data = pd_to_tuple(pd.DataFrame(self.ce_hdbscan_results['labx'], columns=['Label']), 'Label')
        hdbscan_table = pdf_table(hdbscan_data, pdf, width=40, cols=(20, 20))
        logging.info("Add HDBSCAN clustering overview")
        normal_text("Note: -1 label is for the outliers", pdf, italics=True)

        normal_text('Let us see the dendogram of the HDBSCAN clusters.', pdf)
        start_page(pdf)
        pdf_graph(pdf, with_code=False, filename="dendogram_hdbscan.png", x=0, y=10, w=200, h=0)

        pdf.ln(130)
        normal_text("Let us see the distribution of the Agglomeration clusters.", pdf)

        pdf.ln(10)
        agg_data = pd_to_tuple(pd.DataFrame(self.ce_agg_results['labx'], columns=['Label']), 'Label')
        agg_table = pdf_table(agg_data, pdf, width=40, cols=(20, 20))
        logging.info("Add Agglomeration clustering overview")
        normal_text('Let us see the dendogram of the Agglomeration clusters.', pdf)
        start_page(pdf)
        pdf_graph(pdf, with_code=False, filename="dendogram_agg.png", x=0, y=10, w=200, h=0)

        pdf.ln(140)
        h1('Suggested Reads', pdf)

        text = (
            "We will remove the outlier results (those which are classified from HDBSCAN) and show the rest as per the "
            "Agglomeration clustering technique.")
        normal_text(text, pdf)



        clustered_df = self.df[self.df['index'].isin(self.clustering_indices)]
        clustered_df['HDBSCAN Labels'] = self.ce_hdbscan_results['labx']
        clustered_df['AGG Labels'] = self.ce_agg_results['labx']
        clustered_df = clustered_df[clustered_df['HDBSCAN Labels'] != -1]
        # clustered_df.sort_values(by=['AGG Labels'],inplace=True)
        clusters = clustered_df['AGG Labels'].unique()

        for cluster in clusters:
            h2('Cluster ' + str(cluster), pdf)
            for idx, row in clustered_df[clustered_df['AGG Labels'] == cluster].iterrows():
                try:
                    normal_text('Original Index: ' + str(row['index']), pdf)
                    normal_text('Title: ' + row['Title'], pdf)
                    normal_text(row['URL'], pdf)
                except:
                    print(row['index'])
                    normal_text('Original Index: ' + str(row['index']), pdf)
                    normal_text('No Title: ', pdf)
                    normal_text(row['URL'], pdf)
        logging.info("Add suggsted reads")

        pdf.output(report_file_name, 'F')
