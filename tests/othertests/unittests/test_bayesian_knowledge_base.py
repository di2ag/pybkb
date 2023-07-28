import unittest
import random
import copy
random.seed(111)

from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB

class bkbApiTestCase(unittest.TestCase):
    def setUp(self):
        #-- Initialize Empty BKB
        self.bkb_empty = BKB()
        self.goldfish = BKB()
        self.goldfish.load('test_bkb_lib/goldfish.bkb')

    def test_addComponent(self):
        compidx = self.bkb_empty.addComponent('X1')
        self.assertEqual(compidx, 0)

    def test_addComponentState(self):
        compidx = self.bkb_empty.addComponent('X1')
        stateidx = self.bkb_empty.addComponentState(compidx, 'A')
        self.assertEqual(stateidx, 0)

    def test_addSameComponent(self):
        compidx = self.bkb_empty.addComponent('X1')
        compidx = self.bkb_empty.addComponent('X1')
        self.assertEqual(compidx, 0)

    def test_addSameComponentState(self):
        compidx = self.bkb_empty.addComponent('X1')
        stateidx = self.bkb_empty.addComponentState(compidx, 'A')
        compidx = self.bkb_empty.addComponent('X1')
        stateidx = self.bkb_empty.addComponentState(compidx, 'A')
        self.assertEqual(stateidx, 0)

    def test_save(self):
        self.bkb_empty.save('_bkb_empty.bkb')

    def test_eq(self):
        goldfish_copy = copy.deepcopy(self.goldfish)
        self.assertEqual(self.goldfish, goldfish_copy)

    def test_ne_different_number_snodes(self):
        goldfish_copy = copy.deepcopy(self.goldfish)
        random_snodes = random.sample(goldfish_copy._S_nodes, 1)
        random_snode = random_snodes[0]
        goldfish_copy.removeSNode(random_snode)

        self.assertNotEqual(self.goldfish, goldfish_copy)

    def test_ne_different_inodes(self):
        goldfish_copy = copy.deepcopy(self.goldfish)
        random_comp_idx = random.randint(0, goldfish_copy.getNumberComponents())
        random_inode_idx = random.randint(0, goldfish_copy.getNumberComponentINodes(random_comp_idx))
        goldfish_copy.removeComponentState(random_comp_idx, random_inode_idx)
        goldfish_copy.addComponentState(random_comp_idx, 'New Test State')

        self.assertNotEqual(self.goldfish, goldfish_copy)

    def test_ne_different_snode_probabaility(self):
        goldfish_copy = copy.deepcopy(self.goldfish)
        random_snodes = random.sample(goldfish_copy._S_nodes, 1)
        random_snode = random_snodes[0]
        random_snode.probability = -1

        self.assertNotEqual(self.goldfish, goldfish_copy)

    def test_ne_different_snode_tail(self):
        goldfish_copy = copy.deepcopy(self.goldfish)
        random_snodes = random.sample(goldfish_copy._S_nodes, 1)
        random_snode = random_snodes[0]

        #-- Get a random I-node that is not already in snode tail.
        random_comp_idxs = random.sample({i for i in range(goldfish_copy.getNumberComponents())} - {snode_tail_comp for snode_tail_comp, _ in random_snode.tail}, 1)
        random_comp_idx = random_comp_idxs[0]
        random_inode_idx = random.randint(0, goldfish_copy.getNumberComponentINodes(random_comp_idx))

        #-- Put it in the tail
        random_snode.tail.append((random_comp_idx, random_inode_idx))

        self.assertNotEqual(self.goldfish, goldfish_copy)


