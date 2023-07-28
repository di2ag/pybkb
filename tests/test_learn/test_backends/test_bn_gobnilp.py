import unittest
import pickle
import os, sys
import random
import json
import numpy as np
import itertools
random.seed(111)
from collections import defaultdict
from gurobipy import read

from pybkb.learn.backends.bn.gobnilp import BNGobnilpBackend
from pybkb.learn.backends.bkb.gobnilp import BKBGobnilpBackend
from pybkb.scores import MdlEntScoreNode


class BNSLTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.basepath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.basepath, '../../../', 'data/sprinkler.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = pickle.load(f_)
        self.feature_states_map = {fs: idx for idx, fs in self.feature_states}
        self.features = defaultdict(list)
        for f, s in self.feature_states:
            self.features[f].append(s)
        self.data_eff = BKBGobnilpBackend.format_data_for_probs(self.data)
        self.node_encoding_len = np.log2(len(self.features))


    def test_regress_learn_endstage_scores(self):
        # We are using Iris so we can do all parent sets.
        backend = BNGobnilpBackend(
                score='mdl_ent',
                palim=None,
                only='data',
                )
        # Gather scores from backend
        scores, store = backend.learn(
                self.data,
                self.feature_states,
                end_stage='scores',
                )

        # Test 1: Ensure we get a score for each source.
        self.assertEqual(len(scores), len(self.features))

        # Test 2: Ensure we look at the correct number of parents based on palim.
        max_pasets = defaultdict(set)
        for x, pa_scores in scores.items():
            for pa, score in pa_scores.items():
                max_pasets[x].add(len(pa))
        self.assertListEqual(
                [max(values) for _, values in max_pasets.items()],
                [len(self.features) - 1 for _ in range(len(self.features))]
                )

        # Test 3: Run scores independently and compare to regression test to backend
        store1 = None
        for feature in self.features:
            for palim in range(len(self.features)):
                for pa_features in itertools.combinations(set(self.features.keys()) - {feature}, r=palim):
                    score_node = MdlEntScoreNode(
                            feature,
                            self.node_encoding_len,
                            rv_level=True,
                            pa_set=list(pa_features),
                            states=self.features,
                            indices=False,
                            )
                    h1 = score_node._calc_rvlevel_score(self.data, self.feature_states, self.feature_states_map, store1, self.data_eff)
                    self.assertAlmostEqual(-h1, scores[feature][frozenset(list(pa_features))])
        """
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
        """

    def test_bnsl_gobnilp_mdlent(self):
        learner = BNLearner('gobnilp', 'mdl_ent')
        learner.fit(self.data, self.feature_states)
        # Gurobi has nondeterministic behaviour so can not use bkfs directly for regression tests.
        # But we can use the base Gurobi MIP model that is used to solve to ensure the correct problem is built everytime.
        _stdout = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f
        # We have to write the model first so that when we load it back in the Fingerprint is the same. Weird. 
        learner.m.write('/tmp/sprinkler-model.mps')
        m_regress = read(os.path.join(self.basepath, f'regression_files/sprinkler-bn-model.mps'))
        m_loaded = read('/tmp/sprinkler-model.mps')
        self.assertEqual(m_loaded.Fingerprint, m_regress.Fingerprint)
        sys.stdout = _stdout
        f.close()

    def test_better_sprinkler_bkb(self):
        bn_learner = BNLearner('gobnilp', 'mdl_ent')
        bn_learner.fit(self.data, self.feature_states)
        bn_learner.bn.calculate_cpts_from_data(self.data, self.feature_states)
        bn_bkb = bn_learner.bn.make_data_bkb(self.data, self.feature_states, self.srcs)
        bkb_learner = BKBLearner('gobnilp', 'mdl_ent')
        bkb_learner.fit(self.data, self.feature_states, self.srcs)
        self.assertLessEqual(
                bn_bkb.score(
                    self.data,
                    self.feature_states,
                    'mdl_ent',
                    only='data',
                    is_learned=True,
                    ),
                bkb_learner.learned_bkb.score(
                    self.data,
                    self.feature_states,
                    'mdl_ent',
                    only='data',
                    is_learned=True,
                    )
                )

    def test_bnsl_gobnilp_mdlmi(self):
        learner = BNLearner('gobnilp', 'mdl_mi')
        learner.fit(self.data, self.feature_states)
        self.assertEqual(learner.m.learned_bn.bnlearn_modelstring(), '[cloudy][rain|sprinkler][sprinkler|wet_grass][wet_grass|cloudy]')
