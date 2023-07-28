from pybkb.learn.backends import BKBGobnilpBackend
from pybkb.scores import MdlEntScoreNode
import compress_pickle
import numpy as np
import time

start_time = time.time()
with open('/tmp/Breast_fpkm_log.dat', 'rb') as f_:
    data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
#with open('../data/keel/bkb_data/iris-standard_classification-no_missing_values.dat', 'rb') as f_:
#    data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
#with open('../data/keel/bkb_data/census-standard_classification-no_missing_values.dat', 'rb') as f_:
#    data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')

data = data.astype(np.float64)

'''
data_eff = BKBGobnilpBackend.format_data_for_probs(data, True)

feature_states_index_map = {fs: i for i, fs in enumerate(feature_states)}
score = BKBGobnilpBackend.calculate_local_score_static2(
        0,
        data,
        data_eff,
        feature_states,
        1,
        MdlEntScoreNode,
        feature_states_index_map,
        verbose=True,
        num_workers=10,
        )
input(f'Done in {time.time() - start_time} seconds. Now check memory useage...')
'''
