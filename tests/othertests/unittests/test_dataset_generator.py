import unittest
import itertools
import json
import logging
import sys

from pybkb.python_base.learning.dataset_generator import generate_dataset
from pybkb.python_base.utils import InvalidFeatureLevel, calculate_joint_prob_from_data
from pybkb.python_base.reasoning.reasoning import updating

logging.basicConfig(level=logging.INFO)

class TestGenerateDataset(unittest.TestCase):

    def test_complete_info_user_levels(self):
        feature_levels = [3,3,3]
        dataset, _levels = generate_dataset(100, feature_levels, complete_info=True, seed=111)
        # Make sure all feature combos are covered.
        available_levels = [[level for level in range(num_level)] for num_level in _levels]
        counts = {prod: 0 for prod in itertools.product(*available_levels)}
        for example_name, example_dict in dataset.items():
            features = [None for _ in range(len(example_dict))]
            for feature_name, level_name in example_dict.items():
                feature = int(feature_name.split('_')[-1])
                level = int(level_name.split('_')[-1])
                features[feature] = level
            counts[tuple(features)] += 1
        # Check to make sure no count is zero.
        for features, feature_set_count in counts.items():
            self.assertGreater(feature_set_count,0)

    def test_continous_levels(self):
        feature_levels = [3,3,3, None, (5,20)]
        dataset, _ = generate_dataset(100, feature_levels, complete_info=False, seed=111)
        # Check continous values
        for example, feature_dict in dataset.items():
            # Check that their numbers
            self.assertIsInstance(feature_dict['feature_3'], float)
            self.assertIsInstance(feature_dict['feature_4'], float)
            # Check ranges
            self.assertGreaterEqual(feature_dict['feature_3'], 0)
            self.assertLessEqual(feature_dict['feature_3'], 1)
            self.assertGreaterEqual(feature_dict['feature_4'], 5)
            self.assertLessEqual(feature_dict['feature_4'], 20)

    def test_check_error_handling(self):
        # Bad level information specified
        with self.assertRaises(TypeError):
            _,_ = generate_dataset(100, 3)

        # Unable to make complete info in the number examples specified
        with self.assertRaises(ValueError):
            _,_ = generate_dataset(2, [10,10,10,10,10], complete_info=True)
