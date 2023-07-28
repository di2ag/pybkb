import unittest
import copy

from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.common.bayesianKnowledgeBase import BKB_S_node
from pybkb.python_base.fusion.fusion import fuse
from pybkb.python_base.fusion.fusion_collapse import collapse_sources


class bkbFusionTestCase(unittest.TestCase):
    def setUp(self):
        #-- Initialize Empty BKB
        self.bkb_empty = BKB()
        self.fused_struct_1 = BKB().load('test_bkb_lib/fused_struct_1.bkb', use_pickle=True, compress=False) 
        self.fused_struct_2 = BKB().load('test_bkb_lib/fused_struct_2.bkb', use_pickle=True, compress=False)
        self.fused_struct_3 = BKB().load('test_bkb_lib/fused_struct_3.bkb', use_pickle=True, compress=False)

    def test_collapse_fused_structure_1(self):
        truth_col_bkb = BKB().load('test_bkb_lib/collapsed_struct_1.bkb', use_pickle=True, compress=False)

        self.fused_struct_1.makeGraph()
        col_bkb = collapse_sources(self.fused_struct_1)
        col_bkb.makeGraph()
        #col_bkb.save('test_bkb_lib/collapsed_struct_1.bkb', use_pickle=True)

        self.assertEqual(truth_col_bkb, col_bkb)

    def test_collapse_fused_structure_2(self):
        truth_col_bkb = BKB().load('test_bkb_lib/collapsed_struct_2.bkb', use_pickle=True, compress=False)

        self.fused_struct_2.makeGraph()
        col_bkb = collapse_sources(self.fused_struct_2)
        col_bkb.makeGraph()
        #col_bkb.save('test_bkb_lib/collapsed_struct_2.bkb', use_pickle=True)
        self.assertEqual(truth_col_bkb, col_bkb)

    def test_fusionCollapse(self):
        #-- Load in correct answer bkb
        truth_col_bkb = BKB().load('test_bkb_lib/collapsed_struct_3.bkb', use_pickle=True, compress=False)

        #-- Build test
        bkb = BKB()
        compidx1 = bkb.addComponent('X1')
        state_idx1 = bkb.addComponentState(compidx1, 'S1')
        compidx2 = bkb.addComponent('X2')
        state_idx2 = bkb.addComponentState(compidx2, 'S1')

        bkb.addSNode(BKB_S_node(init_component_index=compidx1,
                                 init_state_index=state_idx1, 
                                 init_probability=1))
        bkb.addSNode(BKB_S_node(init_component_index=compidx2,
                                 init_state_index=state_idx2, 
                                 init_probability=1,
                                 init_tail = [(compidx1, state_idx1)]))

        #-- make copy of bkb
        bkb2 = copy.deepcopy(bkb)

        fused_bkb = fuse([bkb, bkb2],
                        [1 for _ in range(2)],
                        ['bkb_{}'.format(i) for i in range(2)])

        col_bkb = collapse_sources(self.fused_struct_3)
        #col_bkb.save('test_bkb_lib/collapsed_struct_3.bkb', use_pickle=True)
        self.assertEqual(truth_col_bkb, col_bkb)
