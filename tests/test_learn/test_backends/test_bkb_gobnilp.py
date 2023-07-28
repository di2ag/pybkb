import unittest
import pickle
import os, sys
import csv
import random
import compress_pickle
import itertools
import numpy as np
from collections import defaultdict
from gurobipy import read
from unittest.mock import patch

from pybkb.learn.backends.bkb.gobnilp import BKBGobnilpBackend
from pybkb.utils.probability import build_feature_state_map
from pybkb.scores import MdlEntScoreNode

class BKBGobnilpBackendTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.basepath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.basepath, '../../../', 'data/iris-standard_classification-no_missing_values.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = compress_pickle.load(f_, compression='lz4')
        self.feature_states_map = build_feature_state_map(self.feature_states)
        self.feature_states_index_map = {fs: idx for idx, fs in enumerate(self.feature_states)}
        self.node_encoding_len = len(self.feature_states)
        self.data_eff = BKBGobnilpBackend.format_data_for_probs(self.data)

    def test_regress_learn_endstage_scores(self):
        # We are using Iris so we can do all parent sets.
        backend = BKBGobnilpBackend(
                score='mdl_ent',
                palim=None,
                only='data',
                )
        # Gather scores from backend
        scores, all_scores, store = backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage='scores',
                )

        # Test 1: Ensure we get a score for each source.
        self.assertEqual(len(scores), len(self.srcs))

        # Test 2: Ensure we look at the correct number of parents based on palim.
        max_pasets = defaultdict(set)
        for (x, pa_set), score in all_scores.items():
            max_pasets[x].add(len(pa_set))
        self.assertListEqual(
                [max(values) for _, values in max_pasets.items()],
                [len(self.feature_states_map)-1 for _ in range(len(self.feature_states))]
                )

        # Test 3: Run scores independently and compare to regression test to backend
        store1 = None
        for x_state_idx, (feature, state) in enumerate(self.feature_states):
            for palim in range(len(self.feature_states_map)):
                for pa_features in itertools.combinations(set(self.feature_states_map.keys()) - {feature}, r=palim):
                    for pa_prod in itertools.product(*[self.feature_states_map[pa] for pa in pa_features]):
                        score_node = MdlEntScoreNode(
                                x_state_idx,
                                self.node_encoding_len,
                                rv_level=False,
                                pa_set=list(pa_prod),
                                states=None,
                                indices=True,
                                )
                        h1, store1 = score_node._calc_instantiated_score(self.data, self.feature_states_index_map, store1, self.data_eff)
                        if (x_state_idx, frozenset(list(pa_prod))) not in all_scores:
                            self.assertEqual(h1, 0)
                        else:
                            self.assertAlmostEqual(-h1, all_scores[(x_state_idx, frozenset(list(pa_prod)))])
        
        # Test 4: Compare to regression scores for the iris dataset
        backend = BKBGobnilpBackend(
                score='mdl_ent',
                palim=None,
                only=None,
                )
        scores, all_scores, store = backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage='scores',
                )
        with open(os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv'), 'r') as f_:
            reader = csv.reader(f_)
            for i, row in enumerate(reader):
                # Remove header row
                if i == 0:
                    continue
                data_idx = int(row[0])
                x_state_idx = int(row[1])
                score = float(row[2])
                pa_set = frozenset([int(r) for r in row[3:]])
                # Check that all scores is consistent
                self.assertAlmostEqual(score, all_scores[(x_state_idx, pa_set)])
                # Check that source scores are consistent (scores are in str's for Gobnilp so need to cast back)
                self.assertAlmostEqual(score, scores[data_idx][str(x_state_idx)][frozenset([str(pa) for pa in pa_set])])
        # Uncomment to write file
        '''
        filepath = os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv')
        for i, (data_idx, _scores) in enumerate(scores.items()):
            if i == 0:
                BKBGobnilpBackend.write_scores_to_file(_scores, data_idx, filepath, open_mode='w')
                continue
            BKBGobnilpBackend.write_scores_to_file(_scores, data_idx, filepath, open_mode='a', write_header=False)
        '''

    def test_regress_learn_no_mp(self):
        # We are using Iris so we can do all parent sets.
        backend = BKBGobnilpBackend(
                score='mdl_ent',
                palim=None,
                only=None,
                )
        # Gather scores from backend
        bkfs, report = backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage=None,
                nsols=1,
                kbest=True,
                )
        # Test 1: Regression test to Gurobi MIP model
        # Gurobi has nondeterministic behaviour so can not use bkfs directly for regression tests.
        # But we can use the base Gurobi MIP model that is used to solve to ensure the correct problem is built everytime.
        _stdout = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f
        for i, m in enumerate(backend.models):
            # We have to write the model first so that when we load it back in the Fingerprint is the same. Weird. 
            m.write('/tmp/iris-model.mps')
            m_regress = read(os.path.join(self.basepath, f'regression_files/model-iris-{i}.mps'))
            m_loaded = read('/tmp/iris-model.mps')
            self.assertEqual(m_loaded.Fingerprint, m_regress.Fingerprint)
        sys.stdout = _stdout
        f.close()
    
    def test_regress_learn_with_mp(self):
        # We are using Iris so we can do all parent sets.
        backend = BKBGobnilpBackend(
                score='mdl_ent',
                palim=None,
                only=None,
                )
        # Gather scores from backend
        bkfs, report = backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage=None,
                nsols=1,
                kbest=True,
                num_workers=10,
                )
        # Test 1: Regression test to Gurobi MIP model
        # Gurobi has nondeterministic behaviour so can not use bkfs directly for regression tests.
        # But we can use the base Gurobi MIP model that is used to solve to ensure the correct problem is built everytime.
        _stdout = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f
        for i, m in enumerate(backend.models):
            # We have to write the model first so that when we load it back in the Fingerprint is the same. Weird. 
            # Uncomment to write new model
            #m.write(os.path.join(self.basepath, f'regression_files/model-iris-{i}.mps'))
            m.write('/tmp/iris-model.mps')
            m_regress = read(os.path.join(self.basepath, f'regression_files/model-iris-{i}.mps'))
            m_loaded = read('/tmp/iris-model.mps')
            self.assertEqual(m_loaded.Fingerprint, m_regress.Fingerprint)
        sys.stdout = _stdout
        f.close()

    def test_regress_learn_beginstage_scores(self):
        # Load in scores
        scores = BKBGobnilpBackend.read_scores_file(
                os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv')
                )
        # We are using Iris so we can do all parent sets.
        backend = BKBGobnilpBackend(
                score='mdl_ent',
                palim=None,
                only=None,
                )
        # Gather scores from backend
        bkfs, report = backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                begin_stage='scores',
                end_stage=None,
                nsols=1,
                kbest=True,
                num_workers=10,
                scores=scores,
                )
        # Test 1: Regression test to Gurobi MIP model
        # Gurobi has nondeterministic behaviour so can not use bkfs directly for regression tests.
        # But we can use the base Gurobi MIP model that is used to solve to ensure the correct problem is built everytime.
        _stdout = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f
        for i, m in enumerate(backend.models):
            # We have to write the model first so that when we load it back in the Fingerprint is the same. Weird. 
            m.write('/tmp/iris-model.mps')
            m_regress = read(os.path.join(self.basepath, f'regression_files/model-iris-{i}.mps'))
            m_loaded = read('/tmp/iris-model.mps')
            self.assertEqual(m_loaded.Fingerprint, m_regress.Fingerprint)
        sys.stdout = _stdout
        f.close()
