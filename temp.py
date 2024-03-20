import itertools

from BeeLitReview import BeeLitReview
review = BeeLitReview()
review.scraped_data_enhancement()
review.encode_with_transformers()
review.calculate_similarity()
review.calculate_TSP()
#%%
import pandas as pd
to_review_df = pd.read_excel('to_review.xlsx')


def exclude_tuple(element_list,thetuple):
    """Return the tuple if none of the elements are present"""
    exclude = True
    for element in element_list:
        if element in thetuple:
            exclude = False
    if exclude:
        return thetuple

def include_tuple(element_list,thetuple):
    """Return the tuple if at least one of the elements is present"""
    exclude = False
    for element in element_list:
        if element in thetuple:
            exclude = True
    return exclude


index_exclude_tuple = list(itertools.pairwise(to_review_df['index']))
index_chosen = to_review_df[to_review_df['Choose']]['index']
index_exclude_tuple_updated  = [x for x in index_exclude_tuple if exclude_tuple(index_chosen,x) is not None]

cluster_similarity_df = review.similarity_df
#since it is a tuple, we need to take into account that pairs are interchangable
cluster_similarity_df['key'] = list(zip(cluster_similarity_df['pair0'], cluster_similarity_df['pair1']))
cluster_similarity_df['key1'] = list(zip(cluster_similarity_df['pair1'], cluster_similarity_df['pair0']))
cluster_similarity_df = cluster_similarity_df[~cluster_similarity_df['key'].isin(index_exclude_tuple_updated) & ~cluster_similarity_df['key1'].isin(index_exclude_tuple_updated) ]

year = 2023
type = ['Article', 'Conference Paper']
read = 10
citation = 1

#get the cluster similarity of those indices under inspection
cluster_similarity_df['Choose'] = cluster_similarity_df['key'].apply(lambda x : include_tuple(index_chosen,x) )
cluster_similarity_df = cluster_similarity_df[cluster_similarity_df['Choose']]

#get the indices of the articles we want to look at
fileter_index = review.df[(review.df['Year']>=year) & (review.df['Text Type'].isin(type)) & (review.df['Citations']>=citation) & (review.df['Reads'] >=read)  ]['index']
cluster_similarity_df['Choose Filter'] = cluster_similarity_df['key'].apply(lambda x : include_tuple(fileter_index,x) )
cluster_similarity_df = cluster_similarity_df[cluster_similarity_df['Choose Filter']]

#the only thing left is the clustering

# 1. create pairs based on the to_review_df index
# 2. remove it from the similarity df
# 3. extract only the chosen articles
# 4. create filters for the date, citations and reads and filter based on those the similarity df
# 5. cluastering? heirarchical? DBSCAN?
