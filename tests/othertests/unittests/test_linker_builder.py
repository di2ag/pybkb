import unittest
import itertools
import json
import logging
import sys
import copy

from pybkb.python_base.learning.bkb_builder import LinkerBuilder
from pybkb.python_base.learning.dataset_generator import generate_dataset
from pybkb.python_base.utils import InvalidFeatureLevel, calculate_joint_prob_from_data
from pybkb.python_base.reasoning.reasoning import updating

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLinkerBuilder(unittest.TestCase):

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

    def _check_result_dicts_are_equal(self, res_dict1, res_dict2):
        # Check if update keys are the same
        self.assertEqual(set(res_dict1.keys()), set(res_dict2.keys()))

        for feature, state_prob1 in res_dict1.items():
            # Make sure states are the same for each feature
            self.assertEqual(set(state_prob1.keys()), set(res_dict2[feature].keys()))
            for state, prob1 in state_prob1.items():
                self.assertAlmostEqual(prob1, res_dict2[feature][state])

    def test_one_feature_prop(self):
        feature_levels = [3,3,2,2,2,2]
        dataset, levels = generate_dataset(30, feature_levels, seed=111)
        builder = LinkerBuilder(dataset)

        ordering = [[
                'feature_2',
                'feature_3',
                'feature_4',
                'feature_5',
                    ]]

        evidence = {
                #"feature_0": "level_2",
                #"feature_1": "level_0",
                #"feature_2": "level_0",
                #"feature_3": "level_0",
                }

        #targets = ["feature_2"]

        bkb = copy.deepcopy(builder.build_bkb(
                ordering=ordering))

        # Iterate through all ordering features and check that order and linked match up
        for target in ordering[0]:
            logger.info('Checking target: {}'.format(target))
            targets = [target]

            # Run BKB Reasoning
            res_order = updating(bkb, evidence, targets)
            res_order_dict = res_order.process_updates()
            #res_order.summary(include_contributions=False)

            #bkb.makeGraph()
            _bkb = copy.deepcopy(bkb)

            # Test link
            feature_properties = {
                    "feature_0": {
                            "op": '==',
                            "value": 'level_0'
                            }
                    }

            bkb_linked = builder.link(feature_properties, _bkb)

            # Run BKB Reasoning
            res_link = updating(bkb_linked, evidence, targets)
            res_link_dict = res_link.process_updates()
            #res_link.summary(include_contributions=False)
            #bkb.makeGraph()
            #bkb_linked.makeGraph()
            # Check that none interpolated features are equal between ordered and linked
            self._check_result_dicts_are_equal(res_order_dict, res_link_dict)
            #bkb.makeGraph()

    def test_two_feature_prop(self):
        feature_levels = [3,3,2,2,2,2]
        dataset, levels = generate_dataset(30, feature_levels, seed=111)
        builder = LinkerBuilder(dataset)

        ordering = [[
                'feature_2',
                'feature_3',
                'feature_4',
                'feature_5',
                    ]]

        evidence = {
                #"feature_0": "level_2",
                #"feature_1": "level_0",
                #"feature_2": "level_0",
                #"feature_3": "level_0",
                }

        #targets = ["feature_2"]

        bkb = copy.deepcopy(builder.build_bkb(
                ordering=ordering))

        # Iterate through all ordering features and check that order and linked match up
        for target in ordering[0]:
            logger.info('Checking target: {}'.format(target))
            targets = [target]

            # Run BKB Reasoning
            res_order = updating(bkb, evidence, targets)
            res_order_dict = res_order.process_updates()
            #res_order.summary(include_contributions=False)

            #bkb.makeGraph()
            _bkb = copy.deepcopy(bkb)

            # Test link
            feature_properties = {
                    "feature_0": {
                            "op": '==',
                            "value": 'level_0'
                            },
                    "feature_1": {
                            "op": '==',
                            "value": 'level_0'
                            }
                    }

            bkb_linked = builder.link(feature_properties, _bkb)

            # Run BKB Reasoning
            res_link = updating(bkb_linked, evidence, targets)
            res_link_dict = res_link.process_updates()
            #res_link.summary(include_contributions=False)
            #bkb.makeGraph()
            #bkb_linked.makeGraph()
            # Check that none interpolated features are equal between ordered and linked
            self._check_result_dicts_are_equal(res_order_dict, res_link_dict)
            #bkb.makeGraph()

    def test_bigram_frequency_interpolation_one_feature_prop(self):
        feature_levels = [3,3,2,2,2,2]
        dataset, levels = generate_dataset(30, feature_levels, seed=111)
        builder = LinkerBuilder(dataset)

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

        evidence = {
                #"feature_0": "level_2",
                #"feature_1": "level_0",
                #"feature_2": "level_0",
                #"feature_3": "level_0",
                }

        #targets = ["feature_2"]

        bkb = copy.deepcopy(builder.build_bkb(
                ordering=ordering,
                interpolation_strategy=interpolation_strategy,
                interpolation_measure=interpolation_measure,
                interpolation_options=interpolation_options))

        # Iterate through all ordering features and check that order and linked match up
        for target in ordering[0]:
            logger.info('Checking target: {}'.format(target))
            targets = [target]

            # Run BKB Reasoning
            res_order = updating(bkb, evidence, targets)
            res_order_dict = res_order.process_updates()
            #res_order.summary(include_contributions=False)

            #bkb.makeGraph()
            _bkb = copy.deepcopy(bkb)

            # Test link
            feature_properties = {
                    "feature_0": {
                            "op": '==',
                            "value": 'level_0'
                            }
                    }

            bkb_linked = builder.link(feature_properties, _bkb)

            # Run BKB Reasoning
            res_link = updating(bkb_linked, evidence, targets)
            res_link_dict = res_link.process_updates()
            #res_link.summary(include_contributions=False)
            #bkb.makeGraph()
            #bkb_linked.makeGraph()
            # Check that none interpolated features are equal between ordered and linked
            self._check_result_dicts_are_equal(res_order_dict, res_link_dict)
            #bkb.makeGraph()

    def test_bigram_frequency_interpolation_two_feature_prop(self):
        feature_levels = [3,3,2,2,2,2]
        dataset, levels = generate_dataset(30, feature_levels, seed=111)
        builder = LinkerBuilder(dataset)

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

        evidence = {
                #"feature_0": "level_2",
                #"feature_1": "level_0",
                #"feature_2": "level_0",
                #"feature_3": "level_0",
                }

        #targets = ["feature_2"]

        bkb = copy.deepcopy(builder.build_bkb(
                ordering=ordering,
                interpolation_strategy=interpolation_strategy,
                interpolation_measure=interpolation_measure,
                interpolation_options=interpolation_options))

        # Iterate through all ordering features and check that order and linked match up
        for target in ordering[0]:
            logger.info('Checking target: {}'.format(target))
            targets = [target]

            # Run BKB Reasoning
            res_order = updating(bkb, evidence, targets)
            res_order_dict = res_order.process_updates()
            #res_order.summary(include_contributions=False)

            #bkb.makeGraph()
            _bkb = copy.deepcopy(bkb)

            # Test link
            feature_properties = {
                    "feature_0": {
                            "op": '==',
                            "value": 'level_0'
                            },
                    "feature_1": {
                            "op": '==',
                            "value": 'level_0'
                            }
                    }

            bkb_linked = builder.link(feature_properties, _bkb)

            # Run BKB Reasoning
            res_link = updating(bkb_linked, evidence, targets)
            res_link_dict = res_link.process_updates()
            #res_link.summary(include_contributions=False)
            #bkb.makeGraph()
            #bkb_linked.makeGraph()
            # Check that none interpolated features are equal between ordered and linked
            self._check_result_dicts_are_equal(res_order_dict, res_link_dict)
            #bkb.makeGraph()

    def test_bigram_frequency_interpolation_continuous_feature(self):
        feature_levels = [None,3,2,2,2,2]
        dataset, levels = generate_dataset(30, feature_levels, seed=111)
        builder = LinkerBuilder(dataset)

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

        evidence = {
                #"feature_0": "level_2",
                #"feature_1": "level_0",
                #"feature_2": "level_0",
                #"feature_3": "level_0",
                }

        #targets = ["feature_2"]

        bkb = copy.deepcopy(builder.build_bkb(
                ordering=ordering,
                interpolation_strategy=interpolation_strategy,
                interpolation_measure=interpolation_measure,
                interpolation_options=interpolation_options))

        # Iterate through all ordering features and check that order and linked match up
        for target in ordering[0]:
            logger.info('Checking target: {}'.format(target))
            targets = [target]

            # Run BKB Reasoning
            res_order = updating(bkb, evidence, targets)
            res_order_dict = res_order.process_updates()
            #res_order.summary(include_contributions=False)

            #bkb.makeGraph()
            _bkb = copy.deepcopy(bkb)

            # Test link
            feature_properties = {
                    "feature_0": {
                            "op": '>=',
                            "value": 0.3
                            },
                    "feature_1": {
                            "op": '==',
                            "value": 'level_0'
                            }
                    }

            bkb_linked = builder.link(feature_properties, _bkb)

            # Run BKB Reasoning
            res_link = updating(bkb_linked, evidence, targets)
            res_link_dict = res_link.process_updates()
            res_link.summary(include_contributions=False)
            #bkb.makeGraph()
            #bkb_linked.makeGraph()
            # Check that none interpolated features are equal between ordered and linked
            self._check_result_dicts_are_equal(res_order_dict, res_link_dict)
            #bkb.makeGraph()

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

