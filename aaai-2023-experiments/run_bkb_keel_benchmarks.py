import os
import ray
from operator import itemgetter

from pybkb.learn import BKBLearner, BNLearner
from pybkb.utils.data import KeelWrangler

# Set datasets path
cwd = os.path.dirname(__file__)
datasets_path = os.path.join(cwd, '../', 'data/keel')

# Dataset Names
datasets = [
        #"adult-standard_classification-with_missing_values.dat",
        #"iris-standard_classification-no_missing_values.dat",
        "mushroom-standard_classification-no_missing_values.dat",
        ]

#datasets = [d for d in os.listdir(datasets_path) if 'no_missing_values' in d]

ray.init(address='auto')

# Sort datasets from easy to hard
bkb_data = []
for dataset in datasets:
    # Make full paths
    path = os.path.join(datasets_path, dataset)
    # Load data
    try:
        wrangler = KeelWrangler(path, 'lz4')
        data, feature_states, srcs = wrangler.get_bkb_dataset(combine_train_test=True)
    except:
        print(f'Error with {dataset}.')
        continue
    bkb_data.append(
            (len(feature_states), data, feature_states, srcs, dataset)
            )
bkb_data = sorted(bkb_data, key=itemgetter(0))

# Run Experiments
for _, data, feature_states, srcs, dataset in bkb_data:
    print('-'*10, f' Running: {dataset} ', '-'*10)
    # Make full paths
    result_path = os.path.join(cwd, 'results', 'bkb', os.path.splitext(dataset)[0])
    # Run learning
    learner = BKBLearner(
            'gobnilp',
            'mdl_ent',
            distributed=True, 
            palim=2,
            )
    # BN learner
    #bn_learner = BNLearner('gobnilp', 'mdl_ent', palim=2)
    #bn_learner.fit(data, feature_states, verbose=True)
    #print(len(bn_learner.backend.store))
    #input('Continue?')
    learner.fit(data, feature_states, verbose=True)
    # Save out results
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # Save learned bkb
    learner.learned_bkb.save(os.path.join(result_path, 'learned.bkb'))
    # Save report
    learner.report.json(filepath=os.path.join(result_path, 'report.json'))
