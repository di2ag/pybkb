import unittest
import itertools
import json
import logging
import sys

from pybkb.python_base.learning.bkb_builder import ChainBuilder
from pybkb.python_base.learning.dataset_generator import generate_dataset
from pybkb.python_base.utils import InvalidFeatureLevel, calculate_joint_prob_from_data
from pybkb.python_base.reasoning.reasoning import updating

logging.basicConfig(level=logging.INFO)

class TestChainBuilder(unittest.TestCase):

    @staticmethod
    def _calculate_actual_priors(bkb, builder, dataset, levels):
        print('All Priors:')
        # Print Priors
        for feature in builder.features_list:
            res = updating(bkb, {}, [feature])
            res.summary(include_contributions=False)
            print('\tActual Prior:')
            for state in builder.features[feature]:
                data_prob = calculate_joint_prob_from_data(dataset, levels, [], (feature, state))
                print('\t\tP({} = {}) = {}'.format(feature, state, data_prob))

    def test_chain_rule_bkb(self):
        dataset, levels = generate_dataset(100, [3,3,3,3], complete_info=True, seed=111)
        builder = ChainBuilder(dataset)
        bkb = builder.build_bkb()

        evidence = {
                "feature_0": "level_0",
                "feature_1": "level_1",
                }

        targets = ["feature_3"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res_dict = res.process_updates()

        # Run Checks
        evidence_direct = [(feature, level) for feature, level in evidence.items()]
        for feature, state_prob in res_dict.items():
            for state, prob in state_prob.items():
                data_prob = calculate_joint_prob_from_data(
                        dataset,
                        levels,
                        evidence_direct,
                        (feature, state)
                        )
                self.assertAlmostEqual(prob,data_prob)

    def test_chain_rule_bkb_with_excluded_feature(self):
        dataset, levels = generate_dataset(100, [3,3,3,3], complete_info=True, seed=111)
        builder = ChainBuilder(dataset)
        bkb = builder.build_bkb(exclude_features=['feature_2'])
        evidence = {
                "feature_0": "level_0",
                "feature_1": "level_1",
                }

        targets = ["feature_3"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res_dict = res.process_updates()

        # Run Checks
        evidence_direct = [(feature, level) for feature, level in evidence.items()]
        for feature, state_prob in res_dict.items():
            for state, prob in state_prob.items():
                data_prob = calculate_joint_prob_from_data(
                        dataset,
                        levels,
                        evidence_direct,
                        (feature, state)
                        )
                self.assertAlmostEqual(prob,data_prob)

    def test_chain_rule_bkb_with_excluded_feature_incomplete_info(self):
        dataset, levels = generate_dataset(10, [3,3,3,3], complete_info=False, seed=111)
        builder = ChainBuilder(dataset)
        bkb = builder.build_bkb(exclude_features=['feature_2'])
        evidence = {
        #        "feature_0": "level_0",
                "feature_1": "level_1",
                }
        #evidence = {}

        targets = ["feature_3"]


        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()
        # Run Checks
        evidence_direct = [(feature, level) for feature, level in evidence.items()]
        for feature, state_prob in res_dict.items():
            for state, prob in state_prob.items():
                data_prob = calculate_joint_prob_from_data(
                        dataset,
                        levels,
                        evidence_direct,
                        (feature, state)
                        )
                if prob < 0:
                    self.assertEqual(data_prob, 0)
                else:
                    self.assertAlmostEqual(prob,data_prob)

    '''
    def test_full_bkb(self):
        dataset, levels = generate_dataset(100, 4, num_levels=[3,3,3,3], complete_info=True, seed=111)
        builder = ChainBuilder(dataset)
        bkb = builder.build_bkb()
        src_evidence = helper_evidence(bkb)
        evidence = {
                "feature_0": "level_0",
                "feature_1": "level_1",
                "-": "--",
                }
        evidence.update(src_evidence)
        bkb.makeGraph()
        print(evidence)

        targets = ["feature_2"]
        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()
    '''

    def _check_result_against_actual_joint_prob(self, res_dict, evidence, targets, dataset, levels, print_joint=False, print_counts=False):
        # Run Checks
        evidence_direct = [(feature, level) for feature, level in evidence.items() if feature != '-']
        for feature, state_prob in res_dict.items():
            for state, prob in state_prob.items():
                data_prob = calculate_joint_prob_from_data(
                        dataset,
                        levels,
                        evidence_direct,
                        (feature, state)
                        )
                if print_joint:
                    print('P({} = {}, E) = {}'.format(feature, state, data_prob))
                    if print_counts:
                        print('Count({} = {}, E) = {}'.format(feature, state, data_prob * len(dataset)))
                    continue
                self.assertAlmostEqual(prob,data_prob)

