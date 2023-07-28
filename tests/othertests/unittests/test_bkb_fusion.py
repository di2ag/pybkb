import unittest
import copy
import os
import random

from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.python_base.fusion.fusion import fuse_from_files

WORKING_DIR = os.getcwd()

class BkbFusion(unittest.TestCase):
    def test_batch_fusion(self):
        bkf_relative_paths = [
               'examples/Fisherman.bkb',
               'examples/IllegalFishingEv.bkb',
               'examples/IllegalDumpingEv.bkb',
               'examples/Pirate.bkb',
               ]
        reliabs = [.5, .8, .2, .3]
        source_names = ['Fisherman', 'IllegalFishingEv', 'IllegalDumpingEv', 'Pirate']
        bkf_files = [os.path.join(WORKING_DIR, '../..', rel_path) for rel_path in bkf_relative_paths]
        
        fusion_no_batch = fuse_from_files(
                bkf_files,
                reliabs,
                source_names,
                verbose=False,
                use_pickle=False,
                )
        fusion_batches = fuse_from_files(
                bkf_files,
                reliabs,
                source_names,
                batch_size=1,
                verbose=False,
                use_pickle=False,
                )
        self.assertEqual(fusion_no_batch, fusion_batches)

    def test_big_batch_fusion(self):
        N = 1000
        bkf_files = [os.path.join(WORKING_DIR, '../../', 'examples/goldfish_binary.bkb') for _ in range(N)]
        reliabs = [random.random() for _ in range(N)]
        source_names = [str(i) for i in range(N)]

        fusion_no_batch = fuse_from_files(
                bkf_files,
                reliabs,
                source_names,
                verbose=True,
                use_pickle=True,
                )
        '''
        fusion_batches = fuse_from_files(
                bkf_files,
                reliabs,
                source_names,
                batch_size=100,
                verbose=False,
                use_pickle=False,
                )
        self.assertEqual(fusion_no_batch, fusion_batches)
        '''
    
