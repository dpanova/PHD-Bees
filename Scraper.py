
# To check
# similarity_df.csv
# df_scraped_articles.xlsx
# path.csv

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
import pandas as pd
from random import randint
from time import sleep
#from selenium_stealth import stealth
import glob
import os
from auxilary_functions import get_file_names
import chromedriver_binary
from selenium.webdriver.common.keys import Keys
#%%

username = "2239079053@shu.bg"
password = "BSru%SQ@trZz&nS26$cJ"
#%%
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
# driver = uc.Chrome(headless=True,use_subprocess=False)
driver = uc.Chrome()
#%%


# driver = webdriver.Chrome()
driver.get("https://www.researchgate.net/login?_sg=ITbsht6C-ko8ZSF49e9deV1BNgFqKApIltdEguj_4uiZ9K_WzI6gYtnTL5xsgogphkn5Z2RJTNuYqd9fMiGJfg")
#%%
uname = driver.find_element("id", "input-login")
uname.send_keys(username)
pword = driver.find_element("id", "input-password")
pword.send_keys(password)

driver.find_element(By.XPATH, '//button[@type="submit"]').click() #TODO update here
directory_path = "/home/deni/Downloads"


#%%

search = driver.find_element('name','query')
search.send_keys('bee acoustic')
search.send_keys(Keys.ENTER)

# driver.find_element(By.XPATH, '//button[@type="submit"]')#TODO update here


driver.find_element(By.CLASS_NAME,'fulltext-availability-switch').click()

#%%
#Get the results from the first page and save them
import pandas as pd
df = pd.DataFrame(columns = ['Title','URL','Authors','Date','Left Category','Right Category','Abstract', 'File Name'])
#%%
result_items = driver.find_elements(By.CLASS_NAME,'search-box__result-item')

for article in result_items:


    # access the inner item stack for each result
    items = article.find_elements(By.CLASS_NAME, 'nova-legacy-v-entity-item__stack-item')
    # get the title from the first tile
    title = items[0].find_element(By.XPATH, 'div//a').text
    print(title)
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
    # authors_list = items[3].find_elements(By.CLASS_NAME, 'nova-legacy-v-entity-item__person-list')
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



    # download
    # click the download button
    # if article.find_elements(By.CLASS_NAME, 'nova-legacy-c-button-group__item')[0].text == 'Download':
    #     article.find_elements(By.CLASS_NAME, 'nova-legacy-c-button-group__item')[0].click()
    #     # find the file name
    #     sleep(randint(30, 60))
    #     list_of_files = get_file_names(directory_path)
    #     # list_of_files = glob.glob('C:/Users/Deni/Downloads')
    #     latest_file = list_of_files[-1]
    # else:
    #     latest_file = 'No file downloaded'

    # append the results to the DF
    page_dict = {'Title': [title],
             'URL': [url],
             'Authors': [author],
             'Date': [date],
             'Left Category': [left],
             'Right Category': [right],
             'Abstract': [abstract]
        # ,
        #      'File': [latest_file]

             }
    page_df = pd.DataFrame(page_dict)
    df = pd.concat([df, page_df], ignore_index=True)
df.to_csv('scraped_articles_18Mar.csv', index=False)

#%%

current_url = driver.current_url
#%%
for step in range(400 , 1000, 10):
    main_url = current_url + '&limit=10&offset=' + str(step)
    driver.get(main_url)
    print(main_url)
    sleep(randint(10, 30))

    # do the same steps
    result_items = driver.find_elements(By.CLASS_NAME, 'search-box__result-item')

    for article in result_items:

        # access the inner item stack for each result
        items = article.find_elements(By.CLASS_NAME, 'nova-legacy-v-entity-item__stack-item')
        # get the title from the first tile
        title = items[0].find_element(By.XPATH, 'div//a').text
        print(title)
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
        # authors_list = items[3].find_elements(By.CLASS_NAME, 'nova-legacy-v-entity-item__person-list')
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
            author = 'No authors found'
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
                     # ,
                     #      'File': [latest_file]

                     }
        page_df = pd.DataFrame(page_dict)
        df = pd.concat([df, page_df], ignore_index=True)
    df.to_csv('scraped_articles_18Mar.csv', index=False)
