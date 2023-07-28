import os
import ray
import logging
import tqdm
import compress_pickle
from operator import itemgetter
from ray.util.placement_group import placement_group

from pybkb.learn import BKBLearner, BNLearner
from pybkb.utils.data import KeelWrangler
from pybkb.utils.mp import MPLogger
from pybkb.utils.probability import build_probability_store

## Script Parameters
NUM_NODES = 15
WORKERS_PER_NODE = 5
# Set datasets path
cwd = os.path.dirname(__file__)
datasets_path = os.path.join(cwd, '../', 'data/keel')

## Functions
def setup_cluster():
    # Initialize Ray
    ray.init(address='auto')
    # Setup placement group and bundles
    bundles = [{"CPU": WORKERS_PER_NODE} for _ in range(NUM_NODES)]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())
    return pg, bundles

## Remote Functions
@ray.remote(num_cpus=1)
def get_scores(dataset_path, scores_path, max_palim):
    # Get dataset name
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    # Setup logger
    logger = MPLogger(dataset_name, logging.INFO, loop_report_time=60)
    # Initialize wrangler and load data
    logger.info('Loading data...')
    wrangler = KeelWrangler(dataset_path, 'lz4')
    # Change palim if it exceeds number of features in the dataset
    max_palim = min(max_palim, len(wrangler.features) - 1)
    data, feature_states, srcs = wrangler.get_bkb_dataset(combine_train_test=True)
    logger.info('Data loaded.')
    # Make full paths
    result_path = os.path.join(scores_path, 'bkb', dataset_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    store = build_probability_store()
    all_scores = {}
    for palim in range(1, max_palim+1):
        # Setup learner
        learner = BKBLearner(
                'gobnilp',
                'mdl_ent',
                distributed=False, 
                palim=palim,
                )
        # Add in all the calculations we've previously done
        learner.backend.store = store
        learner.backend.all_scores = all_scores
        # Collect scores
        logger.info(f'Collecting scores for palim = {palim}...')
        logger.initialize_loop_reporting()
        scores = learner.backend.calculate_all_local_scores(
                data,
                feature_states,
                verbose=False,
                logger=logger,
                reset=False,
                )
        # Add just calculated all scores and store
        store = learner.backend.store
        all_scores = learner.backend.all_scores
        # Save out store and scores
        logger.info('Saving out scores...')
        with open(os.path.join(result_path, f'palim-{palim}.scores'), 'wb') as scores_file:
            compress_pickle.dump(scores, scores_file, compression='lz4')
        logger.info('Scores saved.')
        logger.info('Saving out store.')
        with open(os.path.join(result_path, f'palim-{palim}.store'), 'wb') as store_file:
            compress_pickle.dump(learner.backend.store, store_file, compression='lz4')
    logger.info('Complete.')

# Dataset Names
datasets = [
        #"adult-standard_classification-with_missing_values.dat",
        "iris-standard_classification-no_missing_values.dat",
        "mushroom-standard_classification-no_missing_values.dat",
        #"census-standard_classification-no_missing_values.dat",
        #"penbased-standard_classification-no_missing_values.dat",
        ]
datasets = [d for d in os.listdir(datasets_path) if 'no_missing_values' in d]

"""
# Sort datasets from easy to hard
bkb_data = []
for dataset in datasets:
    if 'with_missing_values' in dataset:
        continue
    # Make full paths
    path = os.path.join(datasets_path, dataset)
    # Load data
    wrangler = KeelWrangler(path, 'lz4')
    _, feature_states, _ = wrangler.get_bkb_dataset(combine_train_test=True)
    # Capture feature states len
    bkb_data.append(
            (len(feature_states), dataset)
            )
# Sort
bkb_data = sorted(bkb_data, key=itemgetter(0))
"""

# Setup logging
logger = MPLogger('Main', logging.INFO)

# Setup Cluster
pg, bundles = setup_cluster()

# Setup score collection work
score_ids = []
bundle_idx = 0
dataset_counter = 0
for dataset in datasets:
    dataset_counter += 1
    if dataset_counter > WORKERS_PER_NODE:
        # Move on to the next bundle
        dataset_counter = 0
        bundle_idx += 1
    # Make full paths
    dataset_path = os.path.abspath(os.path.join(datasets_path, dataset))
    results_path = os.path.abspath(os.path.join(cwd, 'scores'))
    score_ids.append(
            get_scores.options(
                placement_group=pg,
                placement_group_bundle_index=bundle_idx
                ).remote(
                    dataset_path,
                    results_path,
                    10
                    )
                )

i = 0
total = len(score_ids)
logger.initialize_loop_reporting()
while len(score_ids):
    done_ids, score_ids = ray.wait(score_ids)
    _ = ray.get(done_ids[0])
    i += 1
    logger.report(i=i, total=total)
logger.info('Really Complete.')
