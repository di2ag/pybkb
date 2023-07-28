import unittest
import pickle
import os, sys
import random
import json
import numpy as np
random.seed(111)
from gurobipy import read

from pybkb.learn import BNLearner
from pybkb.learn import BKBLearner


class BNSLTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = pickle.load(f_)

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
