from pybkb.common.bayesianKnowledgeBase import bayesianKnowledgeBase as BKB
from pybkb.python_base.reasoning import updating
import time
'''
def process_evidence(evidence_dict, bkb):
    evidence = list()
    for comp_name, state_name in evidence_dict.items():
        comp_idx = bkb.getComponentIndex(comp_name)
        state_idx = bkb.getComponentINodeIndex(comp_idx, state_name)
        evidence.append((comp_idx, state_idx))
    return evidence

def process_targets(target_list, bkb):
    targets = list()
    for target_name in target_list:
        comp_idx = bkb.getComponentIndex(target_name)
        targets.append(comp_idx)
    return targets
'''
if __name__ == '__main__':
    #-- Load in goldfish bkb
    goldfishBkb_path = '/home/cyakaboski/src/python/modules/PyBKB/examples/goldfish.bkb'
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
    '''
    #-- Process Evidence and targets
    evidence = process_evidence(evidence, goldfishBkb)
    targets = process_targets(targets, goldfishBkb)
    '''
    #-- Run Belief Updating
    res = updating(goldfishBkb,
                   evidence,
                   targets,
                   hosts_filename='hosts',
                   num_processes_per_host=10)

    update_time = time.time() - start_time
    res.summary()
    print('Updating Completed in {} sec'.format(update_time))
