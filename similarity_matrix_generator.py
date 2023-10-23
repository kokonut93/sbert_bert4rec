import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from tqdm import tqdm

import pickle

from torchinfo import summary

from sentence_transformers import SentenceTransformer, util
import torch

'''this pyfile generates the similarity matrix based on the below information
1. meta-information of items in Region A (with size n)
2. meta-information of items in Region B (with size m)

Suppose that Region B is Target Region, the aim is to create Similarity Score Matrix
between Region A and B with the size of n X m.
By doing so, we can create matrix where for each item in Region A,
similarity scores for all items across Region B are calculated.



Few dependencies exist.
1. The number of items in both Region A, and B should be considered.


A meta-information dataframe looks like this.
Region A
#################################################################################################################################
||                                                                                                                             ||
||id	        locale	title	                 price	    brand	          color	  size  model   material  author	desc   ||
||B0BCFTG541	JP	    ワタシガモテナイノハドウカ...	600.0	   スクウェア・エニックス   NaN	    NaN	  NaN	  NaN	    タニガワニコ NaN     ||
#################################################################################################################################

Region B

#######################################################################################################################################################
||                                                                                                                                                   ||
||id	        locale	title	                 price	    brand       color	  size  model   material  author	desc                             ||
||09G6JM5WM	ES	Jonami Decoracion Halloween...   12.99	    Jonami	    Naranja	  NaN   NaN     Plástico  NaN	    🎃 ATMÓSFERA MÁXIMA DE HALLOWEEN ||
#######################################################################################################################################################
'''

#1. Load data
dir = './data'
large_locale = 'JP'
small_locale = 'ES'

all_item = pd.read_csv(dir + '/products_train.csv')
large_item = all_item[all_item['locale'] == large_locale]
small_item = all_item[all_item['locale'] == small_locale]
del all_item

with open(dir + f'{large_locale}_unique_values.pkl', 'rb') as f:
    unique_values = pickle.load(f)


#2. Selecting only items in Region A that were bought "above threshold"
# Filter rows based on 'id' column values
large_item_filtered = large_item[large_item['id'].isin(unique_values)]
large_item_filtered = large_item_filtered.reset_index(drop=True)


#3. Processing data - concat title and desc
large_item_filtered['title'] = large_item_filtered['title'].fillna("blank").str.replace('|', '-*-')
small_item['desc'] = small_item['desc'].fillna("blank").str.replace('|', '-')

has_delimiter = large_item_filtered['desc'].str.contains('|', regex=False)
rows_with_delimiter = large_item_filtered['desc'][has_delimiter]
has_delimiter = small_item['desc'].str.contains('|', regex=False)
rows_with_delimiter = small_item['desc'][has_delimiter]

#corpus(Region B) and queries(Region A)
corpus = (small_item['title'] + ' | ' + small_item['desc']).tolist()
queries = (large_item_filtered['title'] + ' | ' + large_item_filtered['desc']).tolist()


#4. Calculate similarity matrix
# Load Sentence Transformer (SBERT)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
#embed corpus
corpus_embedding = model.encode(corpus, convert_to_tensor=True)
top_k = min(200, len(corpus))

# Since the size of matrix is too large to load on RAM, save the matrix on local using memmap
large_to_small_mem = np.memmap(filename = f'./large/{large_locale}_to_{small_locale}_mem', dtype='float32', mode='w+', shape=(len(queries), len(corpus)))
#
for idx, query in enumerate(tqdm(queries)):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
    large_to_small_mem[idx] = cos_scores.cpu().numpy()
#save
large_to_small_mem.flush()

#since there are duplicating items in Region A, and B, the similarity score bewteen the same items is calculated as 1.0
# Find the common values in small_item['id'] and large_item['id']
common_ids = list(set(small_item['id']).intersection(set(large_item_filtered['id'])))
# Print the common values list
print(f'duplicate count : {len(common_ids)}')

large_to_small_duplicate_idx = dict()
for item in tqdm(common_ids):
    large_to_small_duplicate_idx[large_item_filtered[large_item_filtered['id'] == item].index[0]] = small_item[small_item['id'] == item].index[0]
for key, value in large_to_small_duplicate_idx.items():
    large_to_small_mem[key][value] = float(1.0)
#save
large_to_small_mem.flush()


#5. get the "most similar Region B item" for each Region A item
# Iterate through 'large_to_small_mem' and create a dataframe
data = []
for i, row in enumerate(tqdm(large_to_small_mem)):
    item_id = large_item_filtered.loc[i]['id']
    best_small_index = row.argmax()
    similarity_value = np.max(row)
    data.append([i, item_id, best_small_index, similarity_value])

# Create a dataframe with the extracted values
largeid_smallid_score_df = pd.DataFrame(data, columns=['index', 'id', 'best small', 'similarity'])

# Set the 'index' column as the index of the dataframe
largeid_smallid_score_df.set_index('index', inplace=True)

pd.to_csv(dir + '/most_similar_{large_locale}_to_{small_locale}.csv')