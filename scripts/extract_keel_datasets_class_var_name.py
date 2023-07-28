import os
import tqdm
import json
import compress_pickle
from operator import itemgetter

from pybkb.utils.data import KeelWrangler

data_path = '../data/keel'

class_vars = {}
for dataset in tqdm.tqdm(os.listdir(data_path)):
    dataset_name = dataset.split('.')[0]
    if dataset == 'bkb_data':
        continue
    if 'no_missing_values' not in dataset:
        continue
    wrangler = KeelWrangler(os.path.join(data_path, dataset), 'lz4')
    class_vars[dataset_name] = wrangler.predict_class

# Save off these useful files
with open(os.path.join(data_path, 'class_variable_map.json'), 'w') as f_:
    json.dump(class_vars, f_, indent=2)
