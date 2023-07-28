import unittest
import pickle
import os
import random
random.seed(111)

from pybkb.utils.data import KeelWrangler

class DataWranglerTestCase(unittest.TestCase):
    def setUp(self):
        self.data_path = os.path.join(
                os.path.dirname(__file__),
                '../',
                'data/keel',
                )

    def test_keel_adult_missing_values(self):
        wrangler = KeelWrangler(
                os.path.join(self.data_path, 'adult-standard_classification-with_missing_values.dat'),
                compression='lz4',
                )
        data, feature_states, srcs = wrangler.get_bkb_dataset(combine_train_test=True)
        print(data)
        print(feature_states)
        print(data.shape)
