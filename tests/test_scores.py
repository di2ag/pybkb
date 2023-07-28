import unittest
import compress_pickle
import os
import time
import itertools
import numpy as np
from collections import defaultdict

from pybkb.scores import *
from pybkb.learn.backends.bkb.gobnilp import BKBGobnilpBackend

class MdlEntScoreTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open('../data/iris-standard_classification-no_missing_values.dat', 'rb') as f_:
            self.data, self.feature_states, srcs = compress_pickle.load(f_, compression='lz4') 
        # Transform to efficient data structure
        self.data_eff = BKBGobnilpBackend.format_data_for_probs(self.data)
        self.feature_state_map = build_feature_state_map(self.feature_states, no_state_names=True)
        self.node_encoding_len = len(self.feature_states)
        self.node_encoding_len_rv = len(self.feature_state_map)
        self.feature_states_index_map = {fs: idx for idx, fs in enumerate(self.feature_states)}

    def test__calc_instantiated_score(self):
        store1 = None
        store2 = None
        store3 = None
        store4 = None
        for x_state_idx, (feature, state) in enumerate(self.feature_states):
            for palim in range(len(self.feature_state_map)):
                for pa_features in itertools.combinations(set(self.feature_state_map.keys()) - {feature}, r=palim):
                    for pa_prod in itertools.product(*[self.feature_state_map[pa] for pa in pa_features]):
                        score_node = MdlEntScoreNode(
                                x_state_idx,
                                self.node_encoding_len,
                                rv_level=False,
                                pa_set=list(pa_prod),
                                states=None,
                                indices=True,
                                )
                        h1, store1 = score_node._calc_instantiated_score(self.data, self.feature_states_index_map, store1, None)
                        h2, store2 = score_node._calc_instantiated_score(self.data, self.feature_states_index_map, store2, self.data_eff)
                        # Will return a negative value for maximization purposes
                        h3, store3 = score_node.calculate_score(self.data, self.feature_states, store3, self.feature_states_index_map, only='data', data_eff=self.data_eff)
                        h4, store4 = instantiated_conditional_entropy(self.data, x_state_idx, list(pa_prod), store=store4, data_eff=self.data_eff)
                        self.assertAlmostEqual(h1, h4)
                        self.assertAlmostEqual(h2, h4)
                        self.assertAlmostEqual(-h3, h4)
    
    def test__calc_rvlevel_score(self):
        store1 = None
        store2 = None
        store3 = None
        store4 = None
        for feature in self.feature_state_map:
            for palim in range(len(self.feature_state_map)):
                for pa_features in itertools.combinations(set(self.feature_state_map.keys()) - {feature}, r=palim):
                    score_node = MdlEntScoreNode(
                            feature,
                            self.node_encoding_len_rv,
                            rv_level=True,
                            pa_set=list(pa_features),
                            states=self.feature_state_map,
                            indices=False,
                            )
                    # Calculate Number of atomic events represented by the node
                    num_atomic_events = np.prod([len(self.feature_state_map[f]) for f in [feature] + list(pa_features)])
                    # Test
                    h1 = score_node._calc_rvlevel_score(self.data, self.feature_states, self.feature_state_map, store1, None)
                    h2 = score_node._calc_rvlevel_score(self.data, self.feature_states, self.feature_state_map, store2, self.data_eff)
                    score3, _ = score_node.calculate_score(self.data, self.feature_states, store3, feature_states_map=self.feature_state_map, only='data', data_eff=self.data_eff)
                    h4, _ = conditional_entropy(self.data, self.feature_states, feature, list(pa_features), store4, self.data_eff, self.feature_state_map)
                    self.assertAlmostEqual(h1, h4)
                    self.assertAlmostEqual(h2, h4)
                    self.assertAlmostEqual(-score3, h4*num_atomic_events)

    def test__extract_joint_probs(self):
        for x_state_idx, (feature, state) in enumerate(self.feature_states):
            for palim in range(len(self.feature_state_map)):
                for pa_features in itertools.combinations(set(self.feature_state_map.keys()) - {feature}, r=palim):
                    for pa_prod in itertools.product(*[self.feature_state_map[pa] for pa in pa_features]):
                        score_node = MdlEntScoreNode(
                                x_state_idx,
                                self.node_encoding_len,
                                rv_level=False,
                                pa_set=list(pa_prod),
                                states=None,
                                indices=True,
                                )
                        required_joints = set([frozenset([x_state_idx] + list(pa_prod))])
                        required_joints.add(frozenset(list(pa_prod)))
                        self.assertSetEqual(required_joints, score_node._extract_joint_probs())


class MdlMutualInfoScoreTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open('../data/iris-standard_classification-no_missing_values.dat', 'rb') as f_:
            self.data, self.feature_states, srcs = compress_pickle.load(f_, compression='lz4') 
        # Transform to efficient data structure
        self.data_eff = BKBGobnilpBackend.format_data_for_probs(self.data)
        self.feature_state_map = build_feature_state_map(self.feature_states, no_state_names=True)
        self.node_encoding_len = len(self.feature_states)
        self.node_encoding_len_rv = len(self.feature_state_map)
        self.feature_states_index_map = {fs: idx for idx, fs in enumerate(self.feature_states)}

    def test__calc_instantiated_score(self):
        store1 = None
        store2 = None
        store3 = None
        store4 = None
        for x_state_idx, (feature, state) in enumerate(self.feature_states):
            for palim in range(len(self.feature_state_map)):
                for pa_features in itertools.combinations(set(self.feature_state_map.keys()) - {feature}, r=palim):
                    for pa_prod in itertools.product(*[self.feature_state_map[pa] for pa in pa_features]):
                        score_node = MdlMutInfoScoreNode(
                                x_state_idx,
                                self.node_encoding_len,
                                rv_level=False,
                                pa_set=list(pa_prod),
                                states=None,
                                indices=True,
                                )
                        i1, store1 = score_node._calc_instantiated_score(self.data, self.feature_states_index_map, store1, None)
                        i2, store2 = score_node._calc_instantiated_score(self.data, self.feature_states_index_map, store2, self.data_eff)
                        # Will return a negative value for maximization purposes
                        i3, store3 = score_node.calculate_score(self.data, self.feature_states, store3, self.feature_states_index_map, only='data', data_eff=self.data_eff)
                        i4, store4 = instantiated_mutual_info(self.data, x_state_idx, list(pa_prod), store=store4, data_eff=self.data_eff)
                        self.assertAlmostEqual(i1, i4)
                        self.assertAlmostEqual(i1, i4)
                        self.assertAlmostEqual(-i3, i4)
    
    def test__calc_rvlevel_score(self):
        store1 = None
        store2 = None
        store3 = None
        store4 = None
        for feature in self.feature_state_map:
            for palim in range(len(self.feature_state_map)):
                for pa_features in itertools.combinations(set(self.feature_state_map.keys()) - {feature}, r=palim):
                    score_node = MdlMutInfoScoreNode(
                            feature,
                            self.node_encoding_len_rv,
                            rv_level=True,
                            pa_set=list(pa_features),
                            states=self.feature_state_map,
                            indices=False,
                            )
                    # Calculate Number of atomic events represented by the node
                    num_atomic_events = np.prod([len(self.feature_state_map[f]) for f in [feature] + list(pa_features)])
                    # Test
                    i1 = score_node._calc_rvlevel_score(self.data, self.feature_states, self.feature_state_map, store1, None)
                    i2 = score_node._calc_rvlevel_score(self.data, self.feature_states, self.feature_state_map, store2, self.data_eff)
                    score3, _ = score_node.calculate_score(self.data, self.feature_states, store3, feature_states_map=self.feature_state_map, only='data', data_eff=self.data_eff)
                    i4, _ = mutual_info(self.data, self.feature_states, feature, list(pa_features), store4, self.data_eff, self.feature_state_map)
                    self.assertAlmostEqual(i1, i4)
                    self.assertAlmostEqual(i1, i4)
                    self.assertAlmostEqual(-score3, i4*num_atomic_events)

    def test__extract_joint_probs(self):
        for x_state_idx, (feature, state) in enumerate(self.feature_states):
            for palim in range(len(self.feature_state_map)):
                for pa_features in itertools.combinations(set(self.feature_state_map.keys()) - {feature}, r=palim):
                    for pa_prod in itertools.product(*[self.feature_state_map[pa] for pa in pa_features]):
                        score_node = MdlMutInfoScoreNode(
                                x_state_idx,
                                self.node_encoding_len,
                                rv_level=False,
                                pa_set=list(pa_prod),
                                states=None,
                                indices=True,
                                )
                        required_joints = set([frozenset([x_state_idx] + list(pa_prod))])
                        required_joints.add(frozenset(list(pa_prod)))
                        required_joints.add(frozenset([x_state_idx]))
                        self.assertSetEqual(required_joints, score_node._extract_joint_probs())
