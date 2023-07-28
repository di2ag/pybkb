import os
import tqdm
import json
import compress_pickle
import numpy as np
from operator import itemgetter
from sklearn.model_selection import KFold

from pybkb.utils.data import KeelWrangler

data_path = '../data/keel/bkb_data'
FOLDS = 10

for dataset in tqdm.tqdm(os.listdir(data_path)):
    if 'no_missing_values' not in dataset:
        continue
    dataset_name = dataset.split('.')[0]
    with open(os.path.join(data_path, dataset), 'rb') as f_:
        data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
    srcs_np = np.array(srcs)
    kf = KFold(n_splits=10, shuffle=True, random_state=111)
    cross_valid_dir = os.path.join(data_path, 'cross_validation_sets', dataset_name)
    if not os.path.exists(cross_valid_dir):
        os.makedirs(cross_valid_dir)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
        data_train, data_test = data[train_idx], data[test_idx]
        srcs_train, srcs_test = srcs_np[train_idx], srcs_np[test_idx]
        cross_valid_obj = (data_train, data_test, feature_states, list(srcs_train), list(srcs_test))
        with open(os.path.join(cross_valid_dir, f'cross_valid-{fold_idx}.dat'), 'wb') as f_:
            compress_pickle.dump(cross_valid_obj, f_, compression='lz4')
