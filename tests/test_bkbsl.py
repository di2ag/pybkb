import unittest
import pickle
import os
import random
import compress_pickle
#random.seed(111)

from pybkb.learn import BKBLearner


class BKBSLTestCase(unittest.TestCase):
    def setUp(self):
        # Load dataset
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = pickle.load(f_)

    def test_bkbsl_gobnilp_mdlent(self):
        learner = BKBLearner('gobnilp', 'mdl_ent', palim=1)
        learner.fit(self.data, self.feature_states, collapse=True)
        print(learner.report)
        #print(learner.report.bkf_data_scores)
        #print(learner.report.bkf_model_scores)

    def test_bkbsl_gobnilp_mdlmi(self):
        learner = BKBLearner('gobnilp', 'mdl_mi')
        learner.fit(self.data, self.feature_states)
        print(learner.report)
    
    def test_bkbsl_gobnilp_mdlent_distributed(self):
        learner = BKBLearner('gobnilp', 'mdl_ent', palim=2, distributed=True, ray_address='auto')
        learner.fit(self.data, self.feature_states)
        print(learner.report)

    def test_bkbsl_equivalence_mdl(self):
        with open('../data/iris-standard_classification-no_missing_values.dat', 'rb') as f_:
            data, feature_states, srcs = compress_pickle.load(f_, compression='lz4')
        learner_seq = BKBLearner('gobnilp', 'mdl_ent', palim=1)
        learner_dist = BKBLearner('gobnilp', 'mdl_ent', palim=1, distributed=True, ray_address='auto')
        learner_seq.fit(data, feature_states, srcs=srcs)
        learner_dist.fit(data, feature_states, srcs=srcs)
        '''
        print(len(learner_seq.backend.model_strings))
        print(len(learner_dist.backend.model_strings))
        
        for idx, model_string in learner_seq.backend.model_strings.items():
            if model_string != learner_dist.backend.model_strings[idx]:
                print(idx)
                print(model_string)
                print(learner_dist.backend.model_strings[idx])
                input('Continue?')
        with open('scores.test', 'wb') as f_:
            pickle.dump(learner_seq.row_scores, f_)
        for key, scores in learner_seq.row_scores.items():
            if scores != learner_dist.backend.row_scores[key]:
                print(scores)
                print(learner_dist.backend.row_scores[key])
                input('Continue?')
        for key, scores in learner_dist.backend.row_scores.items():
            if scores != learner_seq.row_scores[key]:
                print(scores)
                print(learner_seq.row_scores[key])
                input('Continue?')
        all_scores1 = learner_seq.scores
        all_scores2 = learner_dist.scores
        all_scores1 = {}
        for (x, pa), score in learner_seq.scores.items():
            pa = frozenset([int(p) for p in pa])
            all_scores1[(x, pa)] = score
        for (x, pa), score in learner_dist.scores.items():
            pa = frozenset([int(p) for p in pa])
            all_scores2[(x, pa)] = score
        print(all_scores1)
        print(all_scores2)
        self.assertDictEqual(all_scores1, all_scores2)
        for key, item in all_scores1.items():
            try:
                if all_scores2[key] != item:
                    print(item)
                    print(all_scores2[key])
            except:
                print(f'No: {key}')
        store1 = learner_seq.backend.store
        store2 = learner_dist.backend.store
        store1.pop('__ncalls__')
        store1.pop('__nhashlookups__')
        store2.pop('__ncalls__')
        store2.pop('__nhashlookups__')
        print(store1)
        print(store2)
        self.assertDictEqual(store1, store2)
        '''
        
        print(learner_seq.report.json())
        print(learner_dist.report.json())
        # Tests
        for bkf1, bkf2 in zip(learner_seq.bkfs, learner_dist.bkfs):
            print(bkf1.json())
            print('-'*20)
            print(bkf2.json())
            input('Continue?')
