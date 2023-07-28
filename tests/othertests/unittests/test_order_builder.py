import unittest
import itertools
import json
import logging
import sys

from pybkb.python_base.learning.bkb_builder import OrderBuilder
from pybkb.python_base.learning.dataset_generator import generate_dataset
from pybkb.python_base.utils import InvalidFeatureLevel, calculate_joint_prob_from_data
from pybkb.python_base.reasoning.reasoning import updating

logging.basicConfig(level=logging.INFO)

class TestOrderBuilder(unittest.TestCase):

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


    def test_no_ordering_collapsed_bkb(self):
        dataset, levels = generate_dataset(100, [3,3,3,3], complete_info=True, seed=111)
        builder = OrderBuilder(dataset)
        bkb = builder.build_bkb()

        evidence = {
                "feature_0": "level_0",
                "feature_1": "level_1",
                "-": "--",
                }

        targets = ["feature_2"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res_dict = res.process_updates()
        res.summary(include_contributions=False)

        # Print actual priors of all features.
        self._calculate_actual_priors(bkb, builder, dataset, levels)

        # Check against actual joint probability.
        self._check_result_against_actual_joint_prob(res_dict, evidence, targets, dataset, levels)

        #bkb.makeGraph(save_file='test_file')

    def test_single_ordering_collapsed_bkb(self):
        dataset, levels = generate_dataset(100, [2,2,2,2], complete_info=True, seed=111)
        builder = OrderBuilder(dataset)
        bkb = builder.build_bkb(ordering=['feature_0'])

        evidence = {
                "feature_0": "level_0",
                "feature_1": "level_1",
                }

        targets = ["feature_2"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()

        # Print actual priors of all features.
        self._calculate_actual_priors(bkb, builder, dataset, levels)

        # Check against actual joint probability.
        self._check_result_against_actual_joint_prob(res_dict, evidence, targets, dataset, levels)
        #bkb.makeGraph()

    def test_double_ordering_collapsed_bkb(self):
        dataset, levels = generate_dataset(100, [2,2,2,2], complete_info=True, seed=111)
        builder = OrderBuilder(dataset)
        bkb = builder.build_bkb(ordering=['feature_0', 'feature_1'])

        evidence = {
                #"feature_0": "level_0",
                "feature_1": "level_0",
                }

        targets = ["feature_2"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()

        # Print actual priors of all features.
        self._calculate_actual_priors(bkb, builder, dataset, levels)

        # Check against actual joint probability.
        self._check_result_against_actual_joint_prob(res_dict, evidence, targets, dataset, levels)
        #bkb.makeGraph()

    def test_triple_ordering_collapsed_bkb(self):
        dataset, levels = generate_dataset(100, [2,2,2,2], complete_info=True, seed=111)
        builder = OrderBuilder(dataset)
        bkb = builder.build_bkb(ordering=['feature_0', 'feature_1', 'feature_2'])

        evidence = {
                #"feature_0": "level_0",
                "feature_1": "level_0",
                "feature_2": "level_0",
                }

        targets = ["feature_3"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()

        # Print actual priors of all features.
        self._calculate_actual_priors(bkb, builder, dataset, levels)

        # Check against actual joint probability.
        self._check_result_against_actual_joint_prob(res_dict, evidence, targets, dataset, levels)
        #bkb.makeGraph()

    def test_incomplete_info_collapsed_bkb(self):
        dataset, levels = generate_dataset(10, [3,3,3,3], seed=111)
        builder = OrderBuilder(dataset)
        bkb = builder.build_bkb(ordering=['feature_0'])

        evidence = {
                "feature_0": "level_2",
                "feature_1": "level_0",
                #"feature_2": "level_0",
                }

        targets = ["feature_3"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()

        # Print actual priors of all features.
        self._calculate_actual_priors(bkb, builder, dataset, levels)

        # Check against actual joint probability.
        self._check_result_against_actual_joint_prob(res_dict, evidence, targets, dataset, levels)

    def test_single_ordering_with_multiple_features(self):
        """ This test accesses the capacity to have multiple features on the same order level. Currently,
            only supports single level (more investigation needed). Also, probabilies only match true joint
            when one piece of evidence is specified on the lowest (single) level. This is likely an artifact
            due to the independence conditions, and should also be looked at for future research.
        """
        feature_levels = [3,3,2,2,2,2]
        dataset, levels = generate_dataset(30, feature_levels, seed=111)
        #print(json.dumps(dataset, indent=2))
        builder = OrderBuilder(dataset)
        ordering = [[
                'feature_2',
                'feature_3',
                'feature_4',
                'feature_5',
                    ]]

        bkb = builder.build_bkb(ordering=ordering)

        evidence = {
                #"feature_0": "level_2",
                #"feature_1": "level_0",
                "feature_2": "level_0",
                #"feature_3": "level_0",
                }

        targets = ["feature_0"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()

        # Check against actual joint probability.
        self._check_result_against_actual_joint_prob(res_dict, evidence, targets, dataset, levels)

        #bkb.makeGraph()

    def test_bigram_frequency_interpolation(self):
        feature_levels = [3,3,2,2,2,2]
        dataset, levels = generate_dataset(30, feature_levels, seed=111)
        builder = OrderBuilder(dataset)

        interpolation_strategy = 'bigram'
        interpolation_measure = 'frequency'
        interpolation_options = {
                ('feature_2', 'level_0'): 'level_1',
                ('feature_3', 'level_0'): 'level_1',
                ('feature_4', 'level_0'): 'level_1',
                ('feature_5', 'level_0'): 'level_1',
                }

        ordering = [[
                'feature_2',
                'feature_3',
                'feature_4',
                'feature_5',
                    ]]

        bkb = builder.build_bkb(
                ordering=ordering,
                interpolation_strategy=interpolation_strategy,
                interpolation_measure=interpolation_measure,
                interpolation_options=interpolation_options)

        evidence = {
                #"feature_0": "level_2",
                #"feature_1": "level_0",
                "feature_2": "level_0",
                #"feature_3": "level_0",
                }

        targets = ["feature_0"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)

        #bkb.makeGraph()


    '''
    #TODO:Needs to be looked at and functionality should probably just be removed as it doesn't make a ton of sense yet.

    def test_cooccurring_levels_collapsed_bkb(self):
        num_levels = [2,5]
        max_co_occurring_feature_levels = [1,3]
        dataset, _levels = generate_dataset(10, 2, num_levels=num_levels, max_co_occurring_feature_levels=max_co_occurring_feature_levels, seed=111)
        #print(json.dumps(dataset, indent=2))
        builder = OrderBuilder(dataset)
        bkb = builder.build_bkb()

        evidence = {
                "-": "--",
                #"feature_0": "level_0",
                #"feature_1": "level_0",
                #"feature_2": "level_0",
                }

        targets = ["feature_0"]

        # Run BKB Reasoning
        res = updating(bkb, evidence, targets)
        res.summary(include_contributions=False)
        res_dict = res.process_updates()
        bkb.makeGraph()
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

