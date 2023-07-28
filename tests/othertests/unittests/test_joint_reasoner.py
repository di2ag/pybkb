import unittest
import logging
import json

from pybkb.python_base.learning.dataset_generator import generate_dataset
from pybkb.python_base.reasoning.joint_reasoner import JointReasoner

from pybkb.python_base.learning.test_order_builder import calculate_joint_prob_from_data

logging.basicConfig(level=logging.INFO)

class TestJointReasoner(unittest.TestCase):

    def test_joint_all_categorical(self):
        dataset, levels = generate_dataset(100, [3,3,3,3], complete_info=True, seed=111)
        reasoner = JointReasoner(dataset, None)
        # Set evidence
        evidence = {
            "feature_0": 'level_0',
            "feature_1": 'level_0',
        }
        targets = ['feature_2']
        res, contrib = reasoner.compute_joint(evidence, targets, contribution_features=['feature_3'])
        print(res)
        print(contrib)
        evidence_list = [(feature, level) for feature, level in evidence.items()]
        print(calculate_joint_prob_from_data(dataset, levels, evidence_list, ('feature_2', 'level_1')))

    def test_joint_categorical_and_continuous(self):
        dataset, levels = generate_dataset(100, [None,None,3,3,3,3], seed=111)
        reasoner = JointReasoner(dataset, None)
        # Set evidence
        evidence = {
            "feature_2": 'level_0',
        }
        targets = ['feature_3']
        continuous_evidence = {
            "feature_0": {
                "op": '>=',
                "value": 0.3
            }
        }
        res, contrib = reasoner.compute_joint(
            evidence,
            targets,
            continuous_evidence=continuous_evidence,
        )
        print(res)
        #print(contrib)
        #evidence_list = [(feature, level) for feature, level in evidence.items()]
        #print(calculate_joint_prob_from_data(dataset, levels, evidence_list, ('feature_2', 'level_1')))

    def test_joint_categorical_and_continuous_evidence_and_targets(self):
        dataset, levels = generate_dataset(100, [None,None,3,3,3,3], seed=111)
        reasoner = JointReasoner(dataset, None)
        # Set evidence
        evidence = {
            "feature_2": 'level_0',
        }
        targets = ['feature_3']
        continuous_evidence = {
            "feature_0": {
                "op": '>=',
                "value": 0.3
            }
        }
        continuous_targets = {
            "feature_1": {
                "op": '>=',
                "value": 0.4
            }
        }

        res, contrib = reasoner.compute_joint(
            evidence,
            targets,
            continuous_evidence=continuous_evidence,
            continuous_targets=continuous_targets,
        )
        print(res)
        #print(contrib)
        #evidence_list = [(feature, level) for feature, level in evidence.items()]
        #print(calculate_joint_prob_from_data(dataset, levels, evidence_list, ('feature_2', 'level_1')))
