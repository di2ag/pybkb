from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.cpp_base.fusion import fuse
import time
import os

WORKING_DIR = os.getcwd()

if __name__ == '__main__':
    fishermanBkb = BKB()
    fishermanBkb.load('../examples/Fisherman.bkb')

    illegalFishingBkb = BKB()
    illegalFishingBkb.load('../examples/IllegalFishingEv.bkb')

    illegalDumpingBkb = BKB()
    illegalDumpingBkb.load('../examples/IllegalDumpingEv.bkb')

    pirateBkb = BKB()
    pirateBkb.load('../examples/Pirate.bkb')

    start_time = time.time()
    fuse([fishermanBkb, illegalFishingBkb, illegalDumpingBkb, pirateBkb],
         [.5, .8, .2, .3],
         ['Fisherman', 'IllegalFishingEvent', 'IllegalDumpingEvent', 'Pirate'],
         working_dir=WORKING_DIR)
    fusion_time = time.time() - start_time
    print('Fusion Completed in {} sec'.format(fusion_time))

    fused_bkb = BKB()
    fused_bkb.load('fusion.bkb')

    #Uncomment to make graph
    fused_bkb.makeGraph(layout='neato')
