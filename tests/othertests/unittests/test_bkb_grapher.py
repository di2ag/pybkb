import unittest
import random
import copy
random.seed(111)

from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.python_base.reasoning.reasoning import updating

SHOW = False

class TestBkbGrapher(unittest.TestCase):
    def setUp(self):
        #-- Initialize Goldfish BKB
        self.goldfish = BKB()
        self.goldfish.load('test_bkb_lib/goldfish.bkb')
        #print(self.goldfish.getINodeNames())

    def test_graph_bkb_defaults(self):
        self.goldfish.makeGraph(show=SHOW)
    
    def test_graph_bkb_neato(self):
        self.goldfish.makeGraph(
                show=SHOW,
                layout='neato',
                )
    
    def test_graph_inference(self):
        evidence = {
                "[W] Ammonia 3-Level": 'High',
                "[W] Nitrite Level": '[.5 - 1) ppm',
                }
        target = ['[F] Fish Behavior', '[F] Appetite']

        res = updating(
                self.goldfish,
                evidence,
                target,
                )

        # Iterate through to make sure you can get them all
        for idx in range(res.number_of_inferences('[F] Fish Behavior', 'Scratching against tank objects')):
            inference = res.get_inference(
                    '[F] Fish Behavior',
                    'Scratching against tank objects',
                    idx,
                    )

        # Show the last one
        res.graph_inference(
                inference,
                show=SHOW)
