import unittest
import pickle
import os
import random
import numpy as np
from collections import defaultdict
random.seed(111)

from pybkb.bn import BN
from pybkb.learn import BNLearner
from pybkb.reason.bn import BNReasoner

class BNReasonTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = pickle.load(f_)
        self.states = defaultdict(list)
        for feature, state in self.feature_states:
            self.states[feature].append(state)
        learner = BNLearner('gobnilp', 'mdl_ent', palim=2)
        learner.fit(self.data, self.feature_states)
        self.bn = BN.from_bnlearn_modelstr(learner.bn.bnlearn_modelstring(), self.states)
        self.bn.calculate_cpts_from_data(self.data, self.feature_states)

    def test_to_pomegranate(self):
        reasoner = BNReasoner(self.bn)
        pom_bn = reasoner.to_pomegranate()

    def test_update(self):
        reasoner = BNReasoner(self.bn)
        evidence = {
                'cloudy': 'True',
                'rain': 'True',
                'sprinkler': 'False',
                }
        update = reasoner.update('wet_grass', evidence)
        print(update)

    def test_predict(self):
        reasoner = BNReasoner(self.bn)
        preds = reasoner.predict('wet_grass', self.data[:10,:], self.feature_states)
        print(preds)

    def test_accuracy(self):
        reasoner = BNReasoner(self.bn)
        preds, truths = reasoner.predict('wet_grass', self.data[:100,:], self.feature_states, collect_truths=True)
        print(reasoner.precision_recall_fscore_support(truths, preds, average='weighted'))
