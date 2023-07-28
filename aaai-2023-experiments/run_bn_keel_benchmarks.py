import os
import compress_pickle

from pybkb.learn import BNLearner
from pybkb.utils.data import KeelWrangler

# Set datasets path
cwd = os.path.dirname(__file__)
datasets_path = os.path.join(cwd, '../', 'data/keel')

# Dataset Names
datasets = [
        #"adult-standard_classification-with_missing_values.dat",
        "iris-standard_classification-no_missing_values.dat",
        "mushroom-standard_classification-no_missing_values.dat",
        ]


# Run Experiments
for dataset in datasets:
    print('-'*10, f' Running: {dataset} ', '-'*10)
    # Make full paths
    path = os.path.join(datasets_path, dataset)
    result_path = os.path.join(cwd, 'results', 'bn', os.path.splitext(dataset)[0])
    # Load data
    wrangler = KeelWrangler(path, 'lz4')
    data, feature_states, srcs = wrangler.get_bkb_dataset(combine_train_test=True)
    # Run learning
    learner = BNLearner(
            'gobnilp',
            'mdl_ent',
            palim=2,
            )
    learner.fit(data, feature_states, verbose=True)
    # Save out results
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # Save learned bn
    with open(os.path.join(result_path, 'learned.bn'), 'wb') as f_:
        compress_pickle.dump(learner.bn, f_, compression='lz4')
    # Save report
    learner.report.json(filepath=os.path.join(result_path, 'report.json'))
