import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from tqdm import tqdm
import pickle

#1. Load data
dir = '../data'
large_locale = 'UK'
small_locale = 'ES'

all_session = pd.read_csv(dir + '/sessions_train.csv')
large_session = all_session[all_session['locale'] == large_locale]
del all_session
large_session

#2. Processing as sesion dataset
# Split the 'prev_items' column using space as the delimiter
large_session['prev_items'] = large_session['prev_items'].str.split()
# Remove the brackets from the first and last items
large_session['prev_items'] = large_session['prev_items'].apply(lambda x: [item.strip("[]'") for item in x])
# Create new columns and allocate each string from 'prev_items' to a different column
max_length = large_session['prev_items'].str.len().max()
for i in range(max_length):
    large_session[i] = large_session['prev_items'].str[i]
# Remove the 'prev_items' column
large_session = large_session.drop('prev_items', axis=1)
# Rename the 'next_item' column as 'valid'
large_session = large_session.rename(columns={'next_item': 'valid'})
# Remove the 'locale' column
large_session = large_session.drop('locale', axis=1)
# Move the 'valid' column as the last column
valid_column = large_session['valid']
large_session = large_session.drop('valid', axis=1)
large_session['valid'] = valid_column

#3. Selecting only users with purchase over "threshold"
'''Need to decide on the threshold. The threshold entails "the minimum number of items that the user bought"'''
threshold=10
# Drop rows with NaN values after column "threshold" (in this case, 10)
large_session = large_session.dropna(axis=0, thresh=threshold)
# Reset the index
large_session = large_session.reset_index(drop=True)

# Get unique values from the dataset
unique_values = large_session.values.flatten().tolist()
unique_values = list(set(unique_values))


#4. Save all files
with open(dir + f'/{large_locale}_{threshold}over_unique_item_id.pkl', 'wb') as f:
    pickle.dump(unique_values.index, f)  
large_session.to_csv(dir + f'/{large_locale}_session_{threshold}over.csv', index=False)