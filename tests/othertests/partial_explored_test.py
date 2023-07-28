import os
import random
random.seed(111)

from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.common.bayesianKnowledgeBase import BKB_S_node
from pybkb.cpp_base.reasoning import updating

bkb = BKB()

comp1_idx = bkb.addComponent('G1')
comp1_state_idx = bkb.addComponentState(comp1_idx, 'True')

comp2_idx = bkb.addComponent('G2')
comp2_state_idx = bkb.addComponentState(comp2_idx, 'True')

comp3_idx = bkb.addComponent('G3')
comp3_state_idx = bkb.addComponentState(comp3_idx, 'True')

comp4_idx = bkb.addComponent('S1')
comp4_state_idx = bkb.addComponentState(comp4_idx, 'True')

comp5_idx = bkb.addComponent('S2')
comp5_state_idx = bkb.addComponentState(comp5_idx, 'True')

comp6_idx = bkb.addComponent('S3')
comp6_state_idx = bkb.addComponentState(comp6_idx, 'True')


#Standard Fusion Connections
bkb.addSNode(BKB_S_node(init_component_index=comp4_idx,
                        init_state_index=comp4_state_idx,
                        init_probability=1,
                        init_tail=list()))
bkb.addSNode(BKB_S_node(init_component_index=comp5_idx,
                        init_state_index=comp5_state_idx,
                        init_probability=1,
                        init_tail=list()))
bkb.addSNode(BKB_S_node(init_component_index=comp6_idx,
                        init_state_index=comp6_state_idx,
                        init_probability=1,
                        init_tail=list()))
#bkb.addSNode(BKB_S_node(init_component_index=comp1_idx,
#                        init_state_index=comp1_state_idx,
#                        init_probability=.09,
#                        init_tail=[(comp4_idx, comp4_state_idx)]))
bkb.addSNode(BKB_S_node(init_component_index=comp2_idx,
                        init_state_index=comp2_state_idx,
                        init_probability=.04,
                        init_tail=[(comp5_idx, comp5_state_idx)]))
bkb.addSNode(BKB_S_node(init_component_index=comp3_idx,
                        init_state_index=comp3_state_idx,
                        init_probability=.01,
                        init_tail=[(comp6_idx, comp6_state_idx)]))

#Additionial connections
bkb.addSNode(BKB_S_node(init_component_index=comp1_idx,
                        init_state_index=comp1_state_idx,
                        init_probability=1,
                        init_tail=[(comp2_idx, comp2_state_idx)]))

res = updating(bkb,
               evidence={'G1':'True', 'G2':'True', 'G3':'True'},#, 'G2':'True'},
             targets=['S1', 'S2', 'S3'])
res.summary()
bkb.makeGraph()
