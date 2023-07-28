import unittest
import compress_pickle
import os
import time
import itertools
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

from pybkb.utils.probability import *
from pybkb.learn.backends.bkb.gobnilp import BKBGobnilpBackend
from pybkb.exceptions import InvalidProbabilityError

class ProbabilityTestCase(unittest.TestCase):
    def setUp(self):
        self.wkdir = os.path.dirname(os.path.abspath(__file__))
        with open('../data/iris-standard_classification-no_missing_values.dat', 'rb') as f_:
            self.data, self.feature_states, _ = compress_pickle.load(f_, compression='lz4') 
        # Transform to efficient data structure
        self.data_eff = BKBGobnilpBackend.format_data_for_probs(self.data)
        # Collect feature and states
        fs_dict = defaultdict(list)
        features = set()
        for f, s in self.feature_states:
            fs_dict[f].append(s)
            features.add(f)
        self.fs_dict = dict(fs_dict)
        self.features = list(features)
        self.states = []
        for f in features:
            self.states.append(self.fs_dict[f])
        self.data_df = pd.DataFrame.from_records(self.data, columns=self.feature_states)
        # Generate Test distribution from the following joint
        joint_prob_matrix = np.array([
            [0.1, 0.23, 0.22],
            [0.5, 0, 0.33],
            [0.34, 0.11, 0.03],
            [0.9, 0.05, 0.05],
            ])
        self.joint_prob_matrix = joint_prob_matrix / joint_prob_matrix.sum()
        self.gen_data, self.gen_feature_states, self.gen_data_eff = self.sample_from_known_joint(
                self.joint_prob_matrix,
                )

    def calculate_prob_using_pandas(self, feature_state_names):
        """ Helper function that we know will calculate the right joint prob but is slow.
        """
        joint_probs = self.data_df.value_counts(feature_state_names, normalize=True)
        try:
            joint_prob = joint_probs.loc[tuple([1.0 for _ in range(len(feature_state_names))])]
        except KeyError as ex:
            return 0
        return joint_prob

    @staticmethod
    def sample_from_known_joint(joint_prob_matrix, n=100000):
        samples = np.random.choice(joint_prob_matrix.size, n, p=joint_prob_matrix.ravel())
        # Need to flip array because of the unravel index
        data = np.fliplr(np.vstack(np.unravel_index(samples, joint_prob_matrix.shape)).T)
        # Go thru and create BKB data, feature states and efficient BKB data
        # Collect feature states
        feature_states = set()
        for row in data:
            for feature, state in enumerate(row):
                feature_states.add((feature, state))
        feature_states = list(feature_states)
        fs_map = {fs: idx for idx, fs in enumerate(feature_states)}
        # Encode BKB data
        bkb_data = []
        for row in data:
            enc_row = [0 for _ in range(len(feature_states))]
            for feature, state in enumerate(row):
                enc_row[fs_map[(feature, state)]] = 1
            bkb_data.append(np.array(enc_row))
        bkb_data = np.array(bkb_data)
        # Get data efficiently formatted
        bkb_data_eff = BKBGobnilpBackend.format_data_for_probs(bkb_data)
        return bkb_data.astype(np.float64), feature_states, bkb_data_eff

    def remap_jpm(self, jpm, x_state_indices, parent_state_indices):
        """ Helper function only needed to test joint prob matrix extraction for generated example.
        Function is not needed in the general case.
        """
        # Remap due to feature states mapping being different.
        jpm_remap = np.zeros(jpm.shape)
        for i, fs_i in enumerate(parent_state_indices):
            for j, fs_j in enumerate(x_state_indices):
                remap_i = self.gen_feature_states[fs_i][1]
                remap_j = self.gen_feature_states[fs_j][1]
                jpm_remap[remap_i, remap_j] = jpm[i,j]
        return jpm_remap

    def test_joint_prob_regular(self):
        store = None
        start_time = time.time() 
        for pa_lim in range(len(self.features)):
            # Test simple prior probs
            if pa_lim == 0:
                for idx, fs in enumerate(self.feature_states):
                    prob, store = joint_prob(self.data, idx, store=store)
                    prob_pd = self.calculate_prob_using_pandas([fs])
                    self.assertAlmostEqual(prob, prob_pd)
                continue
            # Test all joint combos
            for feature_indices in itertools.combinations(range(len(self.features)), pa_lim+1):
                states = [self.states[i] for i in feature_indices]
                for prod in itertools.product(*states):
                    fs_prod = [(self.features[f_idx], s) for f_idx, s in zip(feature_indices, prod)]
                    x = fs_prod[0]
                    pa = fs_prod[1:]
                    x_idx = self.feature_states.index(x)
                    pa_indices = [self.feature_states.index(_pa) for _pa in pa]
                    prob, store = joint_prob(self.data, x_idx, pa_indices, store=store)
                    prob_pd = self.calculate_prob_using_pandas(fs_prod)
                    self.assertAlmostEqual(prob, prob_pd)
        #print(f'Time: {time.time() - start_time}')
    
    def test_joint_prob_eff(self):
        store = None
        start_time = time.time() 
        for pa_lim in range(len(self.features)):
            # Test simple prior probs
            if pa_lim == 0:
                for idx, fs in enumerate(self.feature_states):
                    prob, store = joint_prob_eff(self.data_eff, self.data.shape[0], idx, store=store)
                    prob_pd = self.calculate_prob_using_pandas([fs])
                    self.assertAlmostEqual(prob, prob_pd)
                continue
            # Test all joint combos
            for feature_indices in itertools.combinations(range(len(self.features)), pa_lim+1):
                states = [self.states[i] for i in feature_indices]
                for prod in itertools.product(*states):
                    fs_prod = [(self.features[f_idx], s) for f_idx, s in zip(feature_indices, prod)]
                    x = fs_prod[0]
                    pa = fs_prod[1:]
                    x_idx = self.feature_states.index(x)
                    pa_indices = [self.feature_states.index(_pa) for _pa in pa]
                    prob, store = joint_prob_eff(self.data_eff, self.data.shape[0], x_idx, pa_indices, store=store)
                    prob_pd = self.calculate_prob_using_pandas(fs_prod)
                    self.assertAlmostEqual(prob, prob_pd)
        #print(f'Time: {time.time() - start_time}')

    def test_conditional_instantiated_entropy_from_probs(self):
        # Test 1
        p_xp = 0.2
        p_p = 0.3
        p_xgp = p_xp / p_p

        h1 = conditional_instantiated_entropy_from_probs(p_xp, p_p=p_p)
        h2 = conditional_instantiated_entropy_from_probs(p_xp, p_xgp=p_xgp)
        self.assertAlmostEqual(h1, h2)
        self.assertAlmostEqual(h1, 0.11699250014423122)

        # Test 2
        p_xp = 0
        p_p = 0.3
        p_xgp = p_xp / p_p

        h1 = conditional_instantiated_entropy_from_probs(p_xp, p_p=p_p)
        h2 = conditional_instantiated_entropy_from_probs(p_xp, p_xgp=p_xgp)
        self.assertAlmostEqual(h1, h2)
        self.assertAlmostEqual(h1, 0)
        
        # Test 3
        p_xp = 0.2
        p_p = 0

        with self.assertRaises(InvalidProbabilityError):
            _ = conditional_instantiated_entropy_from_probs(p_xp, p_p=p_p)


    def test_conditional_entropy_from_probs(self):
        joint_prob_matrix = np.array([
                [0.1, 0, 0],
                [0.2, 0.3, 0.2],
                [0, 0, 0.2],
                ])
        h = conditional_entropy_from_probs(joint_prob_matrix)
        self.assertAlmostEqual(h, 1.0896596952)
    
    def test_instantiated_conditional_entropy(self):
        feature_states_map = build_feature_state_map(self.gen_feature_states)
        x_state_indices = feature_states_map[0]
        parent_state_indices_list = [feature_states_map[1]]
        
        h0 = conditional_entropy_from_probs(self.joint_prob_matrix)

        # Test with standard data 
        h1 = 0
        store = None
        for x_state_idx in x_state_indices:
            for parent_state_idx in parent_state_indices_list[0]:
                _h1, store = instantiated_conditional_entropy(self.gen_data, x_state_idx, [parent_state_idx], store=store)
                h1 += _h1
        self.assertAlmostEqual(h0, h1, places=2) 
        
        # Test with efficient data
        h2 = 0
        for x_state_idx in x_state_indices:
            for parent_state_idx in parent_state_indices_list[0]:
                _h2, _ = instantiated_conditional_entropy(self.gen_data, x_state_idx, [parent_state_idx], data_eff=self.gen_data_eff)
                h2 += _h2
        self.assertAlmostEqual(h0, h2, places=2) 
        self.assertAlmostEqual(h1, h2)
    
    def test_extract_joint_prob_matrix(self):
        feature_states_map = build_feature_state_map(self.gen_feature_states)

        x_state_indices = feature_states_map[0]
        parent_state_indices_list = [feature_states_map[1]]

        jpm = extract_joint_prob_matrix(
                self.gen_data,
                x_state_indices=x_state_indices,
                parent_state_indices_list=parent_state_indices_list,
            )
        jpm_remap = self.remap_jpm(jpm, x_state_indices, parent_state_indices_list[0])
        for p_true, p_gen in zip(self.joint_prob_matrix.ravel(), jpm_remap.ravel()):
            self.assertAlmostEqual(p_true, p_gen, places=2)

    def test_conditional_entropy(self):
        feature_states_map = build_feature_state_map(self.gen_feature_states)
        x_state_indices = feature_states_map[0]
        parent_state_indices_list = [feature_states_map[1]]
        
        jpm = extract_joint_prob_matrix(
                self.gen_data,
                x_state_indices=x_state_indices,
                parent_state_indices_list=parent_state_indices_list,
            )
        
        h0 = conditional_entropy_from_probs(self.joint_prob_matrix)
        h1 = conditional_entropy_from_probs(jpm)
        h2, fsm = conditional_entropy(self.gen_data, self.gen_feature_states, 0, [1])
        self.assertAlmostEqual(h0, h1, places=2)
        self.assertAlmostEqual(h0, h2, places=2)

    def test_instantiated_mutual_info_from_probs(self):
        # Test 1
        p_x = 0.2
        p_xp = 0.3
        p_p = 0.5

        i1 = instantiated_mutual_info_from_probs(p_x, p_xp, p_p)
        self.assertAlmostEqual(i1, 0.4754887502163468)

        # Test 2
        p_x = 0.2
        p_xp = 0
        p_p = 0.5

        i1 = instantiated_mutual_info_from_probs(p_x, p_xp, p_p)
        self.assertAlmostEqual(i1, 0)
        
        # Test 3
        p_x = 0
        p_xp = 0.1
        p_p = 0.5

        with self.assertRaises(InvalidProbabilityError):
            _ = instantiated_mutual_info_from_probs(p_x, p_xp, p_p)

        # Test 4
        p_x = 0
        p_xp = 0.1
        p_p = 0

        with self.assertRaises(InvalidProbabilityError):
            _ = instantiated_mutual_info_from_probs(p_x, p_xp, p_p)
        
        # Test 5
        p_x = 0.3
        p_xp = 0.1
        p_p = 0

        with self.assertRaises(InvalidProbabilityError):
            _ = instantiated_mutual_info_from_probs(p_x, p_xp, p_p)

    def test_mutual_info_from_probs(self):
        joint_prob_matrix = np.array([
                [0.1, 0, 0],
                [0.2, 0.3, 0.2],
                [0, 0, 0.2],
                ])
        mi = mutual_info_from_probs(joint_prob_matrix)
        self.assertAlmostEqual(mi, 0.48129089923069257)


    def test_instantiated_mutual_info(self):
        feature_states_map = build_feature_state_map(self.gen_feature_states)
        x_state_indices = feature_states_map[0]
        parent_state_indices_list = [feature_states_map[1]]
       
        i0 = mutual_info_from_probs(self.joint_prob_matrix)
        # Test with standard data 
        i1 = 0
        for x_state_idx in x_state_indices:
            for parent_state_idx in parent_state_indices_list[0]:
                _i1, _ = instantiated_mutual_info(self.gen_data, x_state_idx, [parent_state_idx])
                i1 += _i1
        self.assertAlmostEqual(i0, i1, places=2) 

        # Test with efficient data
        i2 = 0
        for x_state_idx in x_state_indices:
            for parent_state_idx in parent_state_indices_list[0]:
                _i2, _ = instantiated_mutual_info(self.gen_data, x_state_idx, [parent_state_idx], data_eff=self.gen_data_eff)
                i2 += _i2
        self.assertAlmostEqual(i0, i2, places=2) 
        self.assertAlmostEqual(i1, i2)

    def test_mutual_info_entropy_relation(self):
        joint_prob_matrix = np.array([
                [0.1, 0, 0],
                [0.2, 0.3, 0.2],
                [0, 0, 0.2],
                ])
        mi = mutual_info_from_probs(joint_prob_matrix)
        conditional_h = conditional_entropy_from_probs(joint_prob_matrix)
        h_X = conditional_entropy_from_probs(np.expand_dims(np.sum(joint_prob_matrix, axis=0), axis=0))
        self.assertAlmostEqual(h_X - conditional_h, mi)

    def test_mutual_info(self):
        feature_states_map = build_feature_state_map(self.gen_feature_states)
        x_state_indices = feature_states_map[0]
        parent_state_indices_list = [feature_states_map[1]]
        
        jpm = extract_joint_prob_matrix(
                self.gen_data,
                x_state_indices=x_state_indices,
                parent_state_indices_list=parent_state_indices_list,
            )
        
        i0 = mutual_info_from_probs(self.joint_prob_matrix)
        i1 = mutual_info_from_probs(jpm)
        i2, fsm = mutual_info(self.gen_data, self.gen_feature_states, 0, [1])
        self.assertAlmostEqual(i0, i1, places=2)
        self.assertAlmostEqual(i0, i2, places=2)

    def test_get_max_joint_sets(self):
        # Test 1: Length of max_joints should equal the number of unique worlds in the dataset
        max_joints = set()
        for row in self.data:
            max_joints.update(get_max_joint_sets(row, len(self.features)))
        self.assertEqual(np.unique(self.data, axis=0).shape[0], len(max_joints))

        # Test 2: Length of max_joints with joint len = 1 should be length of feature states if data all 
        # feature states are represented.
        max_joints = set()
        for row in self.data:
            max_joints.update(get_max_joint_sets(row, 1))
        self.assertEqual(len(self.feature_states), len(max_joints))

    def test_expand_max_joint_set(self):
        max_joints = set()
        for row in self.data:
            max_joints.update(get_max_joint_sets(row, len(self.features)))
        necessary_joints = set()
        for joint in max_joints:
            necessary_joints.update(expand_max_joint_set(joint))
        self.assertEqual(len(necessary_joints), 321)
