import os
import sys
import random
import itertools
from operator import ge, le, eq
random.seed(116)

from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.common.bayesianKnowledgeBase import BKB_I_node, BKB_component, BKB_S_node
from pybkb.python_base.fusion import fuse
from pybkb.python_base.fusion_collapse import collapse_sources
from pybkb.python_base.fusion_collapse_v2 import collapse_sources as collapse_sources_v2
from pybkb.cpp_base.reasoning import updating

PATIENTS = ['Patient{}'.format(i) for i in range(10)]
GENES = ['Gene{}_mut'.format(i) for i in range(10)]
GENE_VARIANTS = ['Variant{}'.format(i) for i in range(2)]

bkfs = list()

#-- Make BKB frags
for j, _ in enumerate(PATIENTS):
    bkf = BKB()

    for gene in GENES:
        #-- Setup Gene i component.
        comp_idx = bkf.addComponent(gene)
        stateTrue_idx = bkf.addComponentState(comp_idx, 'True')

        if random.choice([True, False]):
            bkf.addSNode(BKB_S_node(init_component_index=comp_idx,
                                    init_state_index=stateTrue_idx,
                                    init_probability=1))

            variant_comp_idx = bkf.addComponent(gene + '_Var')
            variant_state_idx = bkf.addComponentState(variant_comp_idx, random.choice(GENE_VARIANTS))
            bkf.addSNode(BKB_S_node(init_component_index=variant_comp_idx,
                                    init_state_index=variant_state_idx,
                                    init_probability=1,
                                    init_tail=[(comp_idx, stateTrue_idx)]))
#    if j == 0:
#        bkf.makeGraph()
    bkfs.append(bkf)

#-- Fuse patients together.
fused_bkb = fuse(bkfs,
                 [1 for _ in range(len(PATIENTS))],
                 [str(hash(patient)) for patient in PATIENTS],
                 working_dir=os.getcwd())

#fused_bkb.makeGraph(layout='neato')
col_bkb = collapse_sources(fused_bkb)
col_bkb.makeGraph(layout='neato')
col_bkb = collapse_sources_v2(fused_bkb)
col_bkb.makeGraph(layout='neato')
