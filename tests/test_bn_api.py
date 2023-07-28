import unittest
import os
import numpy as np
import pickle

from pybkb.bn import BN
from pybkb.exceptions import BKBNotMutexError


class BNApiTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.wkdir, '../', 'data/sprinkler.dat'), 'rb') as data_file:
            self.data, self.feature_states, _ = pickle.load(data_file)

    def test_simple_build(self):
        bn1 = BN()
        rv_map = {
                'A': [1,2],
                1: ['B', 4],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn1.add_rv(comp, states)
        # Add Parents
        bn1.add_parents('A', [1])
        # Add some cpt entries
        bn1.add_cpt_entry('A', 1, .5, [(1, 'B')])
        bn1.add_cpt_entry('A', 2, .3, [(1, 'B')])
        bn1.add_cpt_entry('A', 1, .45, [(1, 4)])
        bn1.add_cpt_entry('A', 2, .7, [(1, 4)])
        bn1.add_cpt_entry(1, 'B', .2)
        bn1.add_cpt_entry(1, 4, .9)
        #bn1.save('test_bn_lib/test_api_bn1.bn')
        # Load saved bkb version
        bn2 = BN.load('test_bn_lib/test_api_bn1.bn')
        # Assert Equal
        self.assertEqual(bn1, bn2)

    def test_bkb_load(self):
        bkb = BN.load('test_bn_lib/test_api_bn1.bn')
    
    def test_eq(self):
        bn1 = BN.load('test_bn_lib/test_api_bn1.bn')
        bn2 = BN.load('test_bn_lib/test_api_bn1.bn')
        self.assertEqual(bn1, bn2)

    def test_neq(self):
        bn1 = BN.load('test_bn_lib/test_api_bn1.bn')
        bn2 = BN.load('test_bn_lib/test_generate_cpt_for_sprinkler.bn')
        self.assertNotEqual(bn1, bn2)

    def test_generate_cpt_for_sprinkler(self):
        bn = BN()
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn.add_rv(comp, states)
        # Add parents
        bn.add_parents('sprinkler', ['cloudy'])
        bn.add_parents('rain', ['cloudy'])
        bn.add_parents('wet_grass', ['sprinkler', 'cloudy'])
        # Use data to calculate CPT
        bn.calculate_cpts_from_data(self.data, self.feature_states)
        bn_true = BN.load(os.path.join(self.wkdir, 'test_bn_lib', 'test_generate_cpt_for_sprinkler.bn'))
        self.assertEqual(bn, bn_true)
    
    def test_make_bkb_and_score(self):
        bn = BN()
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn.add_rv(comp, states)
        # Add parents
        bn.add_parents('sprinkler', ['cloudy'])
        bn.add_parents('rain', ['cloudy'])
        bn.add_parents('wet_grass', ['sprinkler', 'cloudy'])
        # Use data to calculate CPT
        bn.calculate_cpts_from_data(self.data, self.feature_states)
        d_bn, m_bn = bn.score(self.data, self.feature_states, 'mdl_ent', only='both')
        bn_bkb = bn.make_bkb()
        d_bkb, m_bkb = bn_bkb.score(self.data, self.feature_states, 'mdl_ent', only='both')
        self.assertAlmostEqual(d_bn, d_bkb)
        self.assertGreater(m_bn, m_bkb)

    def test_score_for_sprinkler(self):
        bn = BN()
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn.add_rv(comp, states)
        # Add parents
        bn.add_parents('sprinkler', ['cloudy'])
        bn.add_parents('rain', ['cloudy'])
        bn.add_parents('wet_grass', ['sprinkler', 'cloudy'])
        # Use data to calculate CPT
        bn.calculate_cpts_from_data(self.data, self.feature_states)
        d_bn, m_bn = bn.score(self.data, self.feature_states, 'mdl_ent', only='both')
        self.assertAlmostEqual(d_bn, -3.1936626503811016)
        self.assertEqual(m_bn, -17.0)

    def test_from_bnlearn_modelstring(self):
        bn = BN()
        rv_map = {
                "cloudy": ['True', 'False'],
                "sprinkler": ['True', 'False'],
                "rain": ['True', 'False'],
                "wet_grass": ['True', 'False'],
                }
        # Add random variables
        for comp, states in rv_map.items():
            bn.add_rv(comp, states)
        # Add parents
        bn.add_parents('cloudy', ['rain'])
        bn.add_parents('wet_grass', ['sprinkler', 'rain'])
        bn_from_bnlearn = BN.from_bnlearn_modelstr(
                '[cloudy|rain][rain][sprinkler][wet_grass|sprinkler:rain]',
                rv_map,
                )
        self.assertEqual(bn, bn_from_bnlearn)
