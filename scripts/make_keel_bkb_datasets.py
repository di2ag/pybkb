import os
import tqdm
import json
import compress_pickle
from operator import itemgetter

from pybkb.utils.data import KeelWrangler

data_path = '../data/keel'

examples_order = []
feature_states_order = []
for dataset in tqdm.tqdm(os.listdir(data_path)):
    if dataset == 'bkb_data':
        continue
    if 'no_missing_values' not in dataset:
        continue
    wrangler = KeelWrangler(dataset, 'lz4')
    data, feature_states, srcs = wrangler.get_bkb_dataset(combine_train_test=True)
    examples_order.append((data.shape[0], dataset))
    feature_states_order.append((len(feature_states), dataset))
    with open(os.path.join(data_path, f'bkb_data/{dataset}'), 'wb') as f_:
        compress_pickle.dump((data, feature_states, srcs), f_, compression='lz4')

# Sort the examples and feature states order
examples_order = sorted(examples_order, key=itemgetter(0))
feature_states_order = sorted(feature_states_order, key=itemgetter(0))

# Save off these useful files
with open(os.path.join(data_path, 'num_examples_order.json'), 'w') as f_:
    json.dump(examples_order, f_, indent=2)
with open(os.path.join(data_path, 'feature_states_order.json'), 'w') as f_:
    json.dump(feature_states_order, f_, indent=2)

