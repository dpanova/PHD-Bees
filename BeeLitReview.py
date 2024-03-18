import logging

import pandas as pd
from random import randint
from time import sleep
from selenium.webdriver.common.keys import Keys
import undetected_chromedriver
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class BeeLitReview:
    """
    TODO update the description and the validations
    """
    def __init__(self
                 ,logname='BeeLitReview.log'
                 ,already_scraped=True
                 ,scraped_file_name = 'scraped_articles_18Mar.csv'):
        self.scraped_file_name = scraped_file_name
        logging.basicConfig(filename=logname
                            , filemode='a'
                            , format='%(asctime)s %(levelname)s %(message)s'
                            , datefmt='%H:%M:%S'
                            , level=logging.DEBUG)
        if already_scraped:
            self.df = pd.read_csv(scraped_file_name)
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
