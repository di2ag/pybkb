from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.cpp_base.reasoning import revision, updating
import time

if __name__ == '__main__':
    #-- Load in goldfish bkb
    goldfishBkb_path = '../examples/goldfish.bkb'
    goldfishBkb = BKB(name='goldfish')
    goldfishBkb.load(goldfishBkb_path)

    #-- Define evidence
    evidence = {'[W] Clarity': 'Murky',
                '[F] Appetite': 'Limited/None'}

    #-- Define marginal evidence
    marginal_evidence = {
        '[Objects] Cleanliness': {'Clean': .2}
    }

    #-- Define targets
    targets = ['[W] pH Level']

    start_time = time.time()
    #-- Run Belief Revision (MAP)
    res = revision(goldfishBkb,
                   evidence,
                   marginal_evidence=marginal_evidence,
                   targets=targets)

    revision_time = time.time() - start_time
    print('Revision Completed in {} sec'.format(revision_time))
    res.summary()

    start_time = time.time()
    #-- Run Belief Updating
    res = updating(goldfishBkb,
                   evidence,
                   marginal_evidence=marginal_evidence,
                   targets=targets)

    update_time = time.time() - start_time
    print('Updating Completed in {} sec'.format(update_time))
    res.summary(all_rvs=False)
