import unittest
import pickle
import os
import random
import numpy as np
random.seed(111)


from pybkb.learn import BKBLearner
from pybkb.reason.bkb import BKBReasoner
from pybkb.legacy.python_base.reasoning.reasoning import updating

class BKBReasonTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = pickle.load(f_)
        learner = BKBLearner('gobnilp', 'mdl_ent', palim=1)
        learner.fit(self.data, self.feature_states, collapse=True)
        self.bkb = learner.learned_bkb

    def test_complete_fuse_heuristic(self):
        reasoner = BKBReasoner(bkb=self.bkb)
        evidence = {
                'cloudy': 'True',
                'rain': 'True',
                'sprinkler': 'False',
                }
        legacy_update = reasoner.update('wet_grass', evidence)
        log_updates = reasoner.heuristic_update_complete_fused('wet_grass', evidence)
        legacy_updates = legacy_update.process_updates()
        updates = {target_inode: np.exp(log_prob) for target_inode, log_prob in log_updates.items()}
        for target, state_probs in legacy_updates.items():
            for state, prob in state_probs.items():
                self.assertAlmostEqual(prob, updates[(target, state)])

    def test_predict(self):
        reasoner = BKBReasoner(bkb=self.bkb)
        preds, truths = reasoner.predict(
                'wet_grass',
                self.data[:10],
                self.feature_states, 
                collect_truths=True,
                heuristic='fused_with_complete'
                )
        print(preds)

    def test_accuracy(self):
        reasoner = BKBReasoner(bkb=self.bkb)
        preds, truths = reasoner.predict(
                'wet_grass',
                self.data[:100],
                self.feature_states, 
                collect_truths=True,
                heuristic='fused_with_complete'
                )
        print(reasoner.precision_recall_fscore_support(truths, preds, average='weighted'))

