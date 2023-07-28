from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.cpp_base.reasoning import revision, updating
import time

if __name__ == '__main__':
    #-- Load in fused bkbs
    cpp_bkb_path = 'cpp_somalia_fusion.bkb'
    py_v1_bkb_path = 'py_somalia_v1_fusion.bkb'
    py_v2_bkb_path = 'py_somalia_v2_fusion.bkb'

    #-- Define evidence
    evidence = {'(B) Fishing cost is ok': 'Yes',
                '(X) Boarding skills': 'Yes',
                '(B) Has a skiff': 'No'}

    #-- Define marginal evidence
    #marginal_evidence = {
    #    '[Objects] Cleanliness': {'Clean': .2}
    #}

    #-- Define targets
    targets = ['(B) Illegal fishing has occurred']

    print('Evidence:')
    print(evidence)

    print('Targets:')
    print(targets)

    evidence = dict()
    targets=None

    files = [cpp_bkb_path, py_v1_bkb_path, py_v2_bkb_path]
    prefixs = ['cpp-', 'py_v1-', 'py_v2-']
    for f_, prefix in zip(files, prefixs):
        bkb = BKB()
        bkb.load(f_)
        start_time = time.time()

        start_time = time.time()
        #-- Run Belief Updating
        res = updating(bkb,
                       evidence,
                       #marginal_evidence=marginal_evidence,
                       targets=targets,
                       file_prefix=prefix)

        update_time = time.time() - start_time
        print('Updating Completed in {} sec'.format(update_time))
        res.summary(all_rvs=False)
