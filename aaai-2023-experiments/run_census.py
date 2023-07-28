import os
import compress_pickle
import logging

from pybkb.learn import BKBLearner
from pybkb.utils.data import KeelWrangler
from pybkb.utils.mp import MPLogger

path = '../data/keel/census-standard_classification-no_missing_values.dat'
scores_path = '/home/cyakaboski/src/python/projects/chp_developer/PyBKB/nips_experiments/scores/bkb/census-standard_classification-no_missing_values'
wrangler = KeelWrangler(path, compression='lz4')
data, feature_states, srcs = wrangler.get_bkb_dataset(combine_train_test=True)
print(wrangler.features)
print(len(feature_states))

logger = MPLogger('Main', logging.INFO)

learner = BKBLearner('gobnilp', 'mdl_ent', palim=1, distributed=True, ray_address='auto')

# Calculate scores
scores, store = learner.backend.learn(
        data,
        feature_states,
        verbose=True,
        end_stage='scores',
        )

print(len(scores))
print(len(store))

# Save out store and scores
logger.info('Saving out scores...')
with open(os.path.join(scores_path, f'palim-1.scores'), 'wb') as scores_file:
    compress_pickle.dump(scores, scores_file, compression='lz4')
logger.info('Scores saved.')
logger.info('Saving out store.')
with open(os.path.join(scores_path, f'palim-1.store'), 'wb') as store_file:
    compress_pickle.dump(learner.backend.store, store_file, compression='lz4')
# Fit
#learner.fit(data, feature_states, verbose=True)
