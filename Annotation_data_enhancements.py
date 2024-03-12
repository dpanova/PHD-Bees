# The goal of this file is to convert the beeAnnotations.mlf file into a usable table for splitting and labelling the wav files
# Characteristics of the file:
# - it has the start and end second where bees can be heart
# - each file has a descriptive naming convention such as no queen or active day
# the resulting file will be a csv with a structure which will be used in the FFT

# library
import pandas as pd
from auxilary_functions import split_list
from auxilary_functions import file_name_extract

# read the file
annotation_file = 'beeAnnotations.mlf'
with open(annotation_file) as f:
    lines = f.readlines()

lines_df = pd.DataFrame(lines)

# the precessed file consists of one line. each column is seperated by tab, even at pd level
# first we create a list of values for each row
lines_df['all'] = lines_df[0].apply(lambda x: x.split('\t'))
# then we split the list into a separate columns
lines_df = lines_df.apply(lambda x: split_list(x, 'all'), axis=1)

# add column names
lines_df.rename(columns={0: 'start', 1: 'end', 2: 'label'}, inplace=True)
# add the file name to each relevant row
lines_df['file name'] = lines_df.apply(lambda x: file_name_extract(x, 'start', 'end'), axis=1)
lines_df['file name'].ffill(inplace=True)
# remove empty rows, since the original file had rows containing only the file name
lines_df = lines_df[(lines_df['file name'] != '.\n') & (~lines_df['end'].isna())]

# remove new line character
lines_df['file name'] = lines_df['file name'].str.replace('\n', '')
lines_df['label'] = lines_df['label'].str.replace('\n', '')


# this could be a new function
# adding event labels
lines_df['missing queen'] = lines_df['file name'].str.lower().str.contains('|'.join(['missing queen', 'no_queen']))
# lines_df['queen'] = (lines_df['file name'].str.lower().str.contains('queenbee')) & (~lines_df['file name'].str.lower().str.contains('no'))
lines_df['active day'] = lines_df['file name'].str.lower().str.contains('active - day')
lines_df['swarming'] = lines_df['file name'].str.lower().str.contains('swarming')




# change data types
lines_df['start'] = lines_df['start'].astype(float)
lines_df['end'] = lines_df['end'].astype(float)

# add duration of the event
lines_df['duration'] = lines_df['end'] - lines_df['start']

# add the index since it will be easier to refer to the piece of recording later on
lines_df.reset_index(inplace=True)

# save the resulting file to a csv
lines_df.to_csv('beeAnnotations_enhanced.csv', index=False)

#%%
# Here we will do some statistics for the distribution of the bee or no-bee data

# distribution across files
lines_df[['missing queen', 'queen', 'active day', 'swarming', 'file name']].groupby(
    ['missing queen', 'queen', 'active day', 'swarming'], as_index=False).nunique()
# data is balanced for missing queen and queen, we have 15 active days files and only 2 swarming, we don't have inactive days data

# let us see in terms of overall seconds since files do not have equal distribution

lines_df[['missing queen', 'queen', 'active day', 'swarming', 'duration']].groupby(
    ['missing queen', 'queen', 'active day', 'swarming'], as_index=False).sum()
# in terms of hours of recording no queen and queen are somewhat balanced


# maybe let us see what is the distribution in terms of bee and no-bee
lines_df[['label', 'duration']].groupby('label', as_index=False).sum()
# not balanced at all

# TODO: define healthy hive state
