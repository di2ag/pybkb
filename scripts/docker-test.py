import compress_pickle

from pybkb.learn import BKBLearner, BNLearner


with open('data/keel/post-operative-standard_classification-no_missing_values.dat', 'rb') as data_file:
    data, feature_states, srcs = compress_pickle.load(data_file, 'lz4')

bkb_learner = bkb_learner = BKBLearner(backend='gobnilp', score='mdl_ent', distributed=False, palim=2)
bkb_learner.fit(data, feature_states, srcs=srcs, collapse=True)
