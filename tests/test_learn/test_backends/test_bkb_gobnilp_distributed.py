import unittest
import pickle
import os, sys, time, shutil
import csv
import random
import compress_pickle
import itertools
import ray
import numpy as np
import logging
from collections import defaultdict
from gurobipy import read
from unittest.mock import patch

from pybkb.learn.backends.distributed.bkb.gobnilp import BKBGobnilpDistributedBackend
from pybkb.learn.backends.bkb.gobnilp import BKBGobnilpBackend
from pybkb.utils.probability import *
from pybkb.scores import MdlEntScoreNode
from pybkb.utils.mp import MPLogger

ray.init()

class BKBGobnilpDistributedBackendBaseTestCase(unittest.TestCase):
    
    def check_scores_are_almost_equal(self, data, scores1, scores2):
        for row_idx in range(data.shape[0]):
            # Check to make sure all x_state_indices are the same
            try:
                self.assertSetEqual(set(scores1[row_idx]), set(scores2[row_idx]))
            except AssertionError as e:
                print(f'Issue with heads in row index: {row_idx}')
                raise
            for x_state_idx in scores1[row_idx]:
                # Check to make sure all parent sets are the same
                try:
                    self.assertSetEqual(
                            set(scores1[row_idx][x_state_idx]),
                            set(scores2[row_idx][x_state_idx]),
                            )
                except AssertionError as e:
                    print(f'Issue with parent sets in row index: {row_idx}, feature state index: {x_state_idx}.')
                    raise
                for pa_set, score1 in scores1[row_idx][x_state_idx].items():
                    try:
                        # Check to make sure scores are equal
                        self.assertAlmostEqual(score1, scores2[row_idx][x_state_idx][pa_set])
                    except AssertionError as e:
                        print(f'Scores do not match for row index: {row_idx}, feature state index: {x_state_idx}, pa set: {pa_set}.')
                        raise

    def check_dict_almost_equal(self, dict1, dict2):
        # Check to make sure all necessary joints where calculated
        self.assertSetEqual(set(dict1), set(dict2))
        # Make sure all probs match
        for key1, value1 in dict1.items():
            self.assertAlmostEqual(value1, dict2[key1])

class BKBGobnilpDistributedBackendGeneralTestCase(BKBGobnilpDistributedBackendBaseTestCase):
    """ Test case for the "anytime" distributed learning strategy.
    """
    def setUp(self):
        # Load dataset
        self.basepath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.basepath, '../../../', 'data/iris-standard_classification-no_missing_values.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = compress_pickle.load(f_, compression='lz4')
        self.feature_states_map = build_feature_state_map(self.feature_states)
        self.feature_states_index_map = {fs: idx for idx, fs in enumerate(self.feature_states)}
        self.node_encoding_len = np.log2(len(self.feature_states))
        self.data_eff = BKBGobnilpDistributedBackend.format_data_for_probs(self.data)

    def test_setup_cluster(self):
        backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='anytime',
                cluster_allocation=0.25,
                save_dir='/tmp/bkfs',
                )
        pg, bundles = backend.setup_cluster()

        self.assertGreater(backend.available_cpus, 0)
        self.assertGreater(backend.num_cpus, 0)
        self.assertGreater(backend.num_nodes, 0)
        self.assertGreater(backend.available_cpus_per_node, 0)
        self.assertGreater(backend.num_workers_per_node, 0)
        self.assertTrue(backend.num_score_workers_per_worker is None or backend.num_score_workers_per_worker > 1)
        self.assertEqual(len(bundles), backend.num_nodes)
        for b in bundles:
            self.assertEqual(b["CPU"], backend.num_workers_per_node)

    def test_put_data_on_cluster(self):
        backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='anytime',
                cluster_allocation=0.25,
                save_dir='/tmp/bkfs',
                )
        _ = backend.setup_cluster()
        feature_states_index_map = {fs: idx for idx, fs in enumerate(self.feature_states)}
        backend.put_data_on_cluster(self.data, self.feature_states, self.srcs, feature_states_index_map, None, None)
        self.assertIsInstance(backend.data_id, ray.ObjectRef)
        self.assertIsInstance(backend.data_eff_id, ray.ObjectRef)
        self.assertIsInstance(backend.feature_states_id, ray.ObjectRef)
        self.assertIsInstance(backend.feature_states_index_map_id, ray.ObjectRef)
        self.assertIsInstance(backend.srcs_id, ray.ObjectRef)

    def test_split_over_cluster(self):
        backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='anytime',
                cluster_allocation=0.25,
                save_dir='/tmp/bkfs',
                )
        _ = backend.setup_cluster()
        splits = backend.split_data_over_cluster(self.data)
        self.assertEqual(len(splits), backend.num_cpus)
    

class BKBGobnilpDistributedBackendAnytimeTestCase(BKBGobnilpDistributedBackendBaseTestCase):
    """ Test case for the "anytime" distributed learning strategy.
    """
    def setUp(self):
        # Load dataset
        self.basepath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.basepath, '../../../', 'data/iris-standard_classification-no_missing_values.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = compress_pickle.load(f_, compression='lz4')
        self.feature_states_map = build_feature_state_map(self.feature_states)
        self.feature_states_index_map = {fs: idx for idx, fs in enumerate(self.feature_states)}
        self.node_encoding_len = np.log2(len(self.feature_states))
        self.data_eff = BKBGobnilpDistributedBackend.format_data_for_probs(self.data)

    def test_regress_learn_endstage_scores(self):
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='anytime',
                cluster_allocation=0.25,
                )
        reg_backend = BKBGobnilpBackend(
                'mdl_ent',
                palim=None,
                only=None,
                )

        dscores = distributed_backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage='scores',
                )

        rscores, _, _ = reg_backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage='scores',
                )
        self.check_scores_are_almost_equal(self.data, dscores, rscores)
    
    def test_regress_learn_endstage_scores_with_save_dir(self):
        """ These tests should really be done using mock and patches, but it seems ray is causing issues.
        """
        random_temp_save_dir = f'/tmp/bkfs-{time.time()}'
        try:
            distributed_backend = BKBGobnilpDistributedBackend(
                    'mdl_ent',
                    palim=None,
                    only=None,
                    ray_address=None,
                    strategy='anytime',
                    cluster_allocation=0.25,
                    save_dir=random_temp_save_dir,
                    )
            # Test 1: Check that directory was created
            self.assertTrue(os.path.isdir(random_temp_save_dir))
            dscores = distributed_backend.learn(
                    self.data,
                    self.feature_states,
                    self.srcs,
                    end_stage='scores',
                    )
            # Test 2: Check that there are the correct number of scoring files
            self.assertTrue(len(os.listdir(random_temp_save_dir)), self.data.shape[0])
            # Test 3: Ensure all scores match regression files
            regress_scores = BKBGobnilpBackend.read_scores_file(
                    os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv')
                    )
            dscores = dict()
            for score_file in os.listdir(random_temp_save_dir):
                dscores.update(BKBGobnilpBackend.read_scores_file(os.path.join(random_temp_save_dir, score_file)))
            self.check_scores_are_almost_equal(self.data, regress_scores, dscores)
        except Exception as e:
            # Now remove everything
            shutil.rmtree(random_temp_save_dir)
            raise
        # Now remove everything
        shutil.rmtree(random_temp_save_dir)

    def test_learn_endstage_bkfs(self):
        """ Will not perform regression to saved bkfs because gurobi is nondeterministic and not able to set seeds.
        Regression tests on gurobi models however is performed in another test case.
        """
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='anytime',
                cluster_allocation=0.25,
                )
        bkfs, report = distributed_backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage=None,
                )
        self.assertEqual(len(bkfs), self.data.shape[0])

    def test_regress_learn_endstage_bkfs_with_save_dir(self):
        """ These tests should really be done using mock and patches, but it seems ray is causing issues.
        """
        random_temp_save_dir = f'/tmp/bkfs-{time.time()}'
        try:
            distributed_backend = BKBGobnilpDistributedBackend(
                    'mdl_ent',
                    palim=None,
                    only=None,
                    ray_address=None,
                    strategy='anytime',
                    cluster_allocation=0.25,
                    save_dir=random_temp_save_dir,
                    )
            _ = distributed_backend.learn(
                    self.data,
                    self.feature_states,
                    self.srcs,
                    end_stage=None,
                    )
            # Test 1: Make sure correct amount of BKFs are saved
            self.assertEqual(len([f for f in os.listdir(random_temp_save_dir) if '.bkb' in f]), self.data.shape[0])
            
            # Test 2: Make sure correct amount of scores are saved
            self.assertEqual(len([f for f in os.listdir(random_temp_save_dir) if '.csv' in f]), self.data.shape[0])

            # Test 3: Make sure correct amount of gurobi models are saved
            self.assertEqual(len([f for f in os.listdir(random_temp_save_dir) if '.mps' in f]), self.data.shape[0])

            # Test 4: Ensure regression of all gurobi model files
            _stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            for mps_file in [f for f in os.listdir(random_temp_save_dir) if '.mps' in f]:
                split = mps_file.split('-')
                row_idx = int(split[1])
                m_regress = read(os.path.join(self.basepath, f'regression_files/model-iris-{row_idx}.mps'))
                m_test = read(os.path.join(random_temp_save_dir, mps_file))
                try:
                    self.assertEqual(m_regress.Fingerprint, m_test.Fingerprint)
                except AssertionError as e:
                    print(f'Issue with model on data index: {row_idx}.')
                    raise
            sys.stdout = _stdout
            f.close()
        except Exception as e:
            # Now remove everything
            shutil.rmtree(random_temp_save_dir)
            raise
        # Now remove everything
        shutil.rmtree(random_temp_save_dir)


    def test_learn_beginstage_scores(self):
        # Load in scores
        scores = BKBGobnilpBackend.read_scores_file(
                os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv')
                )

        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='anytime',
                cluster_allocation=0.25,
                )

        bkfs, report = distributed_backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                begin_stage='scores',
                scores=scores,
                )
        self.assertEqual(len(bkfs), self.data.shape[0])


class BKBGobnilpDistributedBackendPrecomputeTestCase(BKBGobnilpDistributedBackendBaseTestCase):
    """ Test case for the "anytime" distributed learning strategy.
    """
    def setUp(self):
        # Load dataset
        self.basepath = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(self.basepath, '../../../', 'data/iris-standard_classification-no_missing_values.dat'), 'rb') as f_:
            self.data, self.feature_states, self.srcs = compress_pickle.load(f_, compression='lz4')
        self.feature_states_map = build_feature_state_map(self.feature_states)
        self.feature_states_index_map = {fs: idx for idx, fs in enumerate(self.feature_states)}
        self.node_encoding_len = np.log2(len(self.feature_states))
        self.data_eff = BKBGobnilpDistributedBackend.format_data_for_probs(self.data)
        self.dummy_logger = MPLogger('Dummy Logger', logging.INFO, loop_report_time=60)
    
    def test_collect_necessary_joints(self):
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='precompute',
                cluster_allocation=0.25,
                )
        pg, bundles = distributed_backend.setup_cluster()
        distributed_backend.palim = 4
        necessary_joints_dist = distributed_backend.collect_necessary_joints(self.data, self.dummy_logger, True, pg)

        # Calculate undistributed
        max_joints = set()
        for row in self.data:
            max_joints.update(get_max_joint_sets(row, len(self.feature_states_map)))
        necessary_joints = set()
        for joint in max_joints:
            necessary_joints.update(expand_max_joint_set(joint))
        # Test
        self.assertSetEqual(necessary_joints_dist, necessary_joints)

    def test_construct_probability_store(self):
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='precompute',
                cluster_allocation=0.25,
                )
        pg, bundles = distributed_backend.setup_cluster()
        distributed_backend.palim = 4
        distributed_backend.put_data_on_cluster(self.data, self.feature_states, self.srcs, self.feature_states_index_map, None, None)
        necessary_joints_dist = distributed_backend.collect_necessary_joints(self.data, self.dummy_logger, True, pg)
        store_dist = distributed_backend.construct_probability_store(list(necessary_joints_dist), len(self.data), self.dummy_logger, True, pg)

        # Calculate undistributed
        max_joints = set()
        for row in self.data:
            max_joints.update(get_max_joint_sets(row, len(self.feature_states_map)))
        necessary_joints = set()
        for joint in max_joints:
            necessary_joints.update(expand_max_joint_set(joint))
        store = None
        for joint in necessary_joints:
            _, store = joint_prob_eff(self.data_eff, self.data.shape[0], None, list(joint), store)
        # Check to make sure all necessary joints where calculated
        self.assertSetEqual(set(store_dist), set(store))
        # Make sure all probs match
        for joint, prob in store_dist.items():
            if type(joint) == str:
                continue
            self.assertAlmostEqual(prob, store[joint])

    def test_regress_calculate_necessary_scores(self):
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='precompute',
                cluster_allocation=0.25,
                )
        # Set required attributes
        distributed_backend.palim = 4
        distributed_backend.node_encoding_len = self.node_encoding_len
        pg, bundles = distributed_backend.setup_cluster()
        distributed_backend.put_data_on_cluster(self.data, self.feature_states, self.srcs, self.feature_states_index_map, None, None)
        necessary_joints_dist = distributed_backend.collect_necessary_joints(self.data, self.dummy_logger, True, pg)
        store_dist = distributed_backend.construct_probability_store(list(necessary_joints_dist), len(self.data), self.dummy_logger, True, pg)
        score_collection = distributed_backend.construct_master_score_collection(necessary_joints_dist)
        all_scores = distributed_backend.calculate_necessary_scores(score_collection, store_dist, self.dummy_logger, True, pg)

        # Read in regression scores
        all_scores_truth = dict()
        scores = BKBGobnilpBackend.read_scores_file(os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv'))
        for _, x_pa_scores in scores.items():
            for x, pa_scores in x_pa_scores.items():
                for pa, score in pa_scores.items():
                    if len(pa) == 0:
                        pa = None
                    else:
                        # Recast to ints
                        pa = frozenset([int(p) for p in pa])
                    all_scores_truth[(int(x), pa)] = score
        # Test
        self.check_dict_almost_equal(all_scores_truth, all_scores)

    def test_learn_endstage_store(self):
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='precompute',
                cluster_allocation=0.25,
                )
        # Set required attributes
        distributed_backend.palim = 4
        distributed_backend.node_encoding_len = self.node_encoding_len
        # Calculate store through normal fn
        pg, bundles = distributed_backend.setup_cluster()
        distributed_backend.put_data_on_cluster(self.data, self.feature_states, self.srcs, self.feature_states_index_map, None, None)
        necessary_joints_dist = distributed_backend.collect_necessary_joints(self.data, self.dummy_logger, True, pg)
        store_dist = distributed_backend.construct_probability_store(list(necessary_joints_dist), len(self.data), self.dummy_logger, True, pg)

        # Calculate store through learn
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='precompute',
                cluster_allocation=0.25,
                )
        
        store = distributed_backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage='store',
                )
        self.check_dict_almost_equal(store, store_dist)
        
    def test_regress_learn_endstage_scores(self):
        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='precompute',
                cluster_allocation=0.25,
                )
        scores, store = distributed_backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                end_stage='scores',
                )
        truth_scores = BKBGobnilpBackend.read_scores_file(os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv'))
        self.check_scores_are_almost_equal(self.data, truth_scores, scores)
    
    def test_regress_learn_endstage_bkfs_with_save_dir(self):
        """ These tests should really be done using mock and patches, but it seems ray is causing issues.
        """
        random_temp_save_dir = f'/tmp/bkfs-{time.time()}'
        try:
            distributed_backend = BKBGobnilpDistributedBackend(
                    'mdl_ent',
                    palim=None,
                    only=None,
                    ray_address=None,
                    strategy='precompute',
                    cluster_allocation=0.25,
                    save_dir=random_temp_save_dir,
                    )
            _ = distributed_backend.learn(
                    self.data,
                    self.feature_states,
                    self.srcs,
                    )
            # Test 1: Make sure correct amount of BKFs are saved
            self.assertEqual(len([f for f in os.listdir(random_temp_save_dir) if '.bkb' in f]), self.data.shape[0])
            
            # Test 2: Make sure correct amount of scores are saved
            self.assertEqual(len([f for f in os.listdir(random_temp_save_dir) if '.csv' in f]), self.data.shape[0])

            # Test 3: Make sure correct amount of gurobi models are saved
            self.assertEqual(len([f for f in os.listdir(random_temp_save_dir) if '.mps' in f]), self.data.shape[0])

            # Test 4: Ensure regression of all gurobi model files
            _stdout = sys.stdout
            f = open(os.devnull, 'w')
            sys.stdout = f
            for mps_file in [f for f in os.listdir(random_temp_save_dir) if '.mps' in f]:
                split = mps_file.split('-')
                row_idx = int(split[1])
                m_regress = read(os.path.join(self.basepath, f'regression_files/model-iris-{row_idx}.mps'))
                m_test = read(os.path.join(random_temp_save_dir, mps_file))
                try:
                    self.assertEqual(m_regress.Fingerprint, m_test.Fingerprint)
                except AssertionError as e:
                    print(f'Issue with model on data index: {row_idx}.')
                    raise
            sys.stdout = _stdout
            f.close()
        except Exception as e:
            # Now remove everything
            shutil.rmtree(random_temp_save_dir)
            raise
        # Now remove everything
        shutil.rmtree(random_temp_save_dir)

    def test_learn_beginstage_scores(self):
        # Load in scores
        scores = BKBGobnilpBackend.read_scores_file(
                os.path.join(self.basepath, 'regression_files/iris-bkb-gobnilp-mdlent-regression-scores.csv')
                )

        distributed_backend = BKBGobnilpDistributedBackend(
                'mdl_ent',
                palim=None,
                only=None,
                ray_address=None,
                strategy='precompute',
                cluster_allocation=0.25,
                )

        bkfs, report = distributed_backend.learn(
                self.data,
                self.feature_states,
                self.srcs,
                begin_stage='scores',
                scores=scores,
                )
        self.assertEqual(len(bkfs), self.data.shape[0])
