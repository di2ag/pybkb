import argparse
import os
import compress_pickle

from pybkb.learn import BKBLearner

def load_data(data_path):
    with open(data_path, 'rb') as datafile:
        return compress_pickle.load(datafile, compression='lz4')

def load_scores(scores_path):
    return load_data(scores_path)

def load_store(store_path):
    return load_data(scores_path)

def run_learning(
        data_path:str,
        palim:int=1,
        score:str='mdl_ent',
        distributed:bool=False,
        ray_address:str='auto',
        scores_path:str=None,
        store_path:str=None,
        results_path:str='',
        collapse:bool=True,
        verbose:bool=True,
        strategy:str='precompute',
        save_bkf_dir:str=None,
        cluster_allocation:float=0.25,
        nsols=1,
        kbest=False,
        mec=False,
        nopruning=False,
        clustering_algo=None,
        ):
    # Load data
    data, feature_states, srcs = load_data(data_path)
    # Load scores, if passed
    if scores_path is not None:
        scores = load_scores(scores_path)
    else:
        scores = None
    # Load store, if passed
    if store_path is not None:
        store = load_store(store_path)
    else:
        store = None
    # Setup learner
    learner = BKBLearner(
            'gobnilp',
            score,
            palim=palim,
            distributed=distributed,
            ray_address=ray_address,
            strategy=strategy,
            save_bkf_dir=save_bkf_dir,
            cluster_allocation=cluster_allocation,
            clustering_algo=clustering_algo,
            )
    # Fit
    learner.fit(
            data,
            feature_states,
            srcs,
            verbose=verbose,
            collapse=collapse,
            scores=scores,
            store=store,
            nsols=nsols,
            kbest=kbest,
            mec=mec,
            pruning=not nopruning
            )
    try:
        # Save learned bkb
        learner.learned_bkb.save(os.path.join(results_path, 'learned.bkb'))
        # Save report
        learner.report.json(filepath=os.path.join(results_path, 'report.json'))   
    except AttributeError:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='Path to data file to fit.')
    parser.add_argument('--palim', default=1, type=int, help='Max parent set limit.')
    parser.add_argument('--score', default='mdl_ent', help='Name of scoring function to use.')
    parser.add_argument('--distributed', action='store_true', default=False, help='Whether to fit distrbuted using ray.')
    parser.add_argument('--ray_address', default='auto', help='Ray address.')
    parser.add_argument('--scores_path', default=None, help='Path to pre-saved scores.')
    parser.add_argument('--store_path', default=None, help='Path to pre-saved scores.')
    parser.add_argument('--results_path', default='', help='Directory where results should be saved.')
    parser.add_argument('--no_collapse', default=False, action='store_true', help='Whether to collapse fused BKB.')
    parser.add_argument('--no_verbose', default=False, action='store_true',  help='Whether to show logs.')
    parser.add_argument('--strategy', default='precompute', help='Distrbution strategy. Whether to "precompute probabilities and scores and then learn, or to learn "anytime".')
    parser.add_argument('--save_bkf_dir', default=None, help='Directory where BKFs will be saved for "anytime" strategy.')
    parser.add_argument('--cluster_allocation', default=0.25, type=float, help='Percentange of ray cluster to allocate.')
    parser.add_argument('--nsols', default=1, type=int, help='Number of BKF solutions per data instance.')
    parser.add_argument('--kbest', default=False, action='store_true', help='Whether set of BKBs should be a set of the best scoring ones.')
    parser.add_argument('--mec', default=False, action='store_true', help='Only one solution per Markov Equivalence Class.')
    parser.add_argument('--no_pruning', default=False, action='store_true', help='Prune solution space, turn on if you want more than one solution.')
    parser.add_argument('--clustering_algo', default=None, help='Clustering algo to use to divide data over the cluster.')
    args = parser.parse_args()

    run_learning(
            args.datapath,
            args.palim,
            args.score,
            args.distributed,
            args.ray_address,
            args.scores_path,
            args.store_path,
            args.results_path,
            not args.no_collapse,
            not args.no_verbose,
            args.strategy,
            args.save_bkf_dir,
            args.cluster_allocation,
            args.nsols,
            args.kbest,
            args.mec,
            args.no_pruning,
            args.clustering_algo,
            )
