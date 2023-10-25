import pandas as pd

from tqdm import tqdm
import pickle


dir = '../data'
large_locale = 'JP'
small_locale = 'ES'


#1. Load data
large_session = pd.read_csv(dir + f'/{large_locale}_session_10over.csv')
largeid_smallid_score_df = pd.read_csv(dir + f'/most_similar_{large_locale}_to_{small_locale}.csv')
all_item = pd.read_csv(dir + '/products_train.csv')

# Create a dictionary mapping 'id' values to 'best small' values
id_best_small_map = dict(zip(largeid_smallid_score_df['id'], largeid_smallid_score_df['best small']))
for col in tqdm(large_session.columns):
    large_session[col] = large_session[col].map(id_best_small_map)


# 2. Generating Region A Dataset by mapping unique ids to original product ID
large_session = large_session.dropna(subset=['valid'], how='all')
for col in large_session.columns:
    large_session[col] = pd.to_numeric(large_session[col], errors='coerce').astype('Int64')
large_session.reset_index(drop=True, inplace=True)

mapped_train = {}
for index, row in large_session.iterrows():
    values = [val for val in row.values if not pd.isna(val)]
    mapped_train[index] = values[:-1]  # Exclude the 'valid' column value

mapped_valid = {}
for index, value in large_session['valid'].items():
    mapped_valid[index] = [value]

with open(dir + f'/{large_locale}_{small_locale}mapped_session_train_10over.pkl', 'wb') as f:
    pickle.dump(mapped_train, f)  

with open(dir + f'/{large_locale}_{small_locale}mapped_session_valid_10over.pkl', 'wb') as f:
    pickle.dump(mapped_valid, f)


# 3. Mapping unique values on (Target) Region B Dataset (both train and test)
all_session = pd.read_csv(dir + '/sessions_train.csv')
small_session = all_session[all_session['locale'] == small_locale].reset_index(drop=True)
del all_session

# Split the 'prev_items' column using space as the delimiter
small_session['prev_items'] = small_session['prev_items'].str.split()
# Remove the brackets from the first and last items
small_session['prev_items'] = small_session['prev_items'].apply(lambda x: [item.strip("[]'") for item in x])
# Create new columns and allocate each string from 'prev_items' to a different column
max_length = small_session['prev_items'].str.len().max()
for i in range(max_length):
    small_session[i] = small_session['prev_items'].str[i]
# Remove the 'prev_items' column
small_session = small_session.drop('prev_items', axis=1)
# Rename the 'next_item' column as 'valid'
small_session = small_session.rename(columns={'next_item': 'valid'})
# Remove the 'locale' column
small_session = small_session.drop('locale', axis=1)
# Move the 'valid' column as the last column
valid_column = small_session['valid']
small_session = small_session.drop('valid', axis=1)
small_session['valid'] = valid_column

# Create a dictionary mapping 'id' values to 'best small' values
small_item = all_item[all_item['locale'] == small_locale]
del all_item

small_id_to_idx = {k:v for v, k in small_item['id'].to_dict().items()}
for col in tqdm(small_session.columns):
    small_session[col] = small_session[col].map(small_id_to_idx)


# Mapping unique ids to original product ID

mapped_train = {}
for index, row in small_session.iterrows():
    values = [val for val in row.values if not pd.isna(val)]
    mapped_train[index] = values[:-1]  # Exclude the 'valid' column value

mapped_valid = {}
for index, value in small_session['valid'].items():
    mapped_valid[index] = [value]

with open(dir + f'/{small_locale}mapped_test_session_train.pkl', 'wb') as f:
    pickle.dump(mapped_train, f)  

with open(dir + f'/{small_locale}mapped_test_session_valid.pkl', 'wb') as f:
    pickle.dump(mapped_valid, f)