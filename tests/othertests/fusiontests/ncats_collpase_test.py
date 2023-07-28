import os
import sys
import random
import itertools
from operator import ge, le, eq
import matplotlib
matplotlib.use('tkagg')
random.seed(116)

from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.common.bayesianKnowledgeBase import BKB_I_node, BKB_component, BKB_S_node
from pybkb.python_base.fusion import fuse
from pybkb.python_base.fusion_collapse import collapse_sources
from pybkb.python_base.reasoning import updating, checkMutex


fused_bkb = BKB()
fused_bkb.load('/home/public/data/ncats/BabelBKBs/smallProblem/fusion.bkb')
print(checkMutex(fused_bkb))
#print(fused_bkb.to_str())
fused_bkb.makeGraph()
col_bkb = collapse_sources(fused_bkb)
#input('Done')

print(checkMutex(col_bkb))
col_bkb.save('collapsed_bkb.bkb')
col_bkb.makeGraph()
