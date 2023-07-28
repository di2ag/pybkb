from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.python_base.fusion.fusion import fuse
import time
import os

WORKING_DIR = os.getcwd()

if __name__ == '__main__':
    fishermanBkb = BKB()
    fishermanBkb.load(os.path.join(WORKING_DIR, '../..', 'examples/Fisherman.bkb'))

    illegalFishingBkb = BKB()
    illegalFishingBkb.load(os.path.join(WORKING_DIR, '../..', 'examples/IllegalFishingEv.bkb'))

    illegalDumpingBkb = BKB()
    illegalDumpingBkb.load(os.path.join(WORKING_DIR, '../..', 'examples/IllegalDumpingEv.bkb'))

    pirateBkb = BKB()
    pirateBkb.load(os.path.join(WORKING_DIR, '../..', 'examples/Pirate.bkb'))

    start_time = time.time()
    fused_bkb = fuse([fishermanBkb, illegalFishingBkb, illegalDumpingBkb, pirateBkb],
         [.5, .8, .2, .3],
         ['Fisherman', 'IllegalFishingEvent', 'IllegalDumpingEvent', 'Pirate'],
         working_dir=WORKING_DIR)
    fusion_time = time.time() - start_time
    print('Fusion Completed in {} sec'.format(fusion_time))
    
    #Uncomment to make graph
    #fused_bkb.makeGraph(layout='neato')
