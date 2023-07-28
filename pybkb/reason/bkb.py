import copy
import itertools
import numpy as np
import tqdm
from collections import defaultdict

from pybkb.bkb import BKB
from pybkb.reason import BaseReasoner

class BKBReasoner(BaseReasoner):
    def __init__(
            self,
            bkb=None,
            filepath:str=None,
            compression:str='lz4',
            sparse_array_format:str='dok',
            ):
        if bkb is None and filepath is None:
            raise ValueError('Must pass a BKB object or filepath to saved BKB.')
        if bkb is not None:
            self.bkb = bkb
        else:
            self.bkb = BKB.load(filepath, compression, sparse_array_format)
        super().__init__()

    def update(self, target:str, evidence:dict, timeout:int=None):
        """ General update function. Currently in development.
        """
        raise NotImplementedError

    def _get_all_supported_snodes(self, head_inode, evidence, snodes_by_head, target_feature=None):
        possible_snodes = []
        for snode_idx in snodes_by_head[head_inode]:
            # Check the tail
            tail = self.bkb.get_snode_tail(snode_idx)
            snode_supported = True
            for tail_feature, tail_state in tail:
                if tail_feature == target_feature and target_feature is not None:
                    continue
                elif '__Source__' in tail_feature:
                    continue
                elif evidence[tail_feature] != tail_state:
                    snode_supported = False
                    break
            if snode_supported:
                possible_snodes.append(snode_idx)
        return possible_snodes


    def heuristic_update_complete_fused(self, target_feature:str, evidence:dict, verbose=0):
        """ A heuristic update reasoning code to be used on fused, i.e. learned BKBs,
            when complete evidence is specified, with a single target RV update. Mainly used
            in testing cross validation accuracy and other such use cases.
        """
        # Initialize updates
        updates = {
                (target_feature, target_state): None for target_state in self.bkb.inodes_map[target_feature]
                }
        # Get snodes by head
        snodes_by_head = self.bkb.snodes_by_head
        # Go thru each target instantiation and make sure that it has an S-node with a supported tail
        possible_target_snodes = {}
        for target_feature, target_state in tqdm.tqdm(updates, desc='Extracting possible S-nodes', leave=False, disable=verbose==0):
            target_evidence = copy.deepcopy(evidence)
            # Add target instantiation to evidence copy
            target_evidence[target_feature] = target_state
            # Go through each evidence I-node and try to find S-nodes that are supported by the evidence and target instantiation
            supported_snodes = {}
            for inode in target_evidence.items():
                possible_snodes = self._get_all_supported_snodes(inode, target_evidence, snodes_by_head)
                # If there wasn't any supported S-nodes for this random variable then there can't be an inference
                if len(possible_snodes) == 0:
                    supported_snodes = None
                    break
                supported_snodes[inode] = possible_snodes
            if supported_snodes is not None:
                possible_target_snodes[(target_feature, target_state)] = supported_snodes
        # Extract S-node options
        for target_inode, supported_snodes in tqdm.tqdm(possible_target_snodes.items(), desc='Processing Updates', leave=False, disable=verbose==0):
            # If we found some valid inferences, now we can just run all combos of sources and sum
            inode_evid_map = {}
            snode_options = []
            for i, (evid_inode, snode_opts) in enumerate(supported_snodes.items()):
                inode_evid_map[evid_inode] = i
                snode_options.append(snode_opts)
            total = np.prod([len(opt) for opt in snode_options])
            for snodes_config in tqdm.tqdm(itertools.product(*snode_options), desc='Processing S-node configs', total=total, leave=False, disable=verbose<1):
                # Collect source I-nodes and target feature instantiations in this config
                source_snodes = []
                for snode_idx in snodes_config:
                    for tail_feature, tail_state in self.bkb.get_snode_tail(snode_idx):
                        if '__Source__' in tail_feature:
                            # There will only be one
                            source_snodes.append(snodes_by_head[(tail_feature, tail_state)][0])
                # Take log sums
                log_prob = sum(
                        [
                            np.log(
                                self.bkb.snode_probs[snode_idx]
                                ) for snode_idx in list(snodes_config) + source_snodes
                            ]
                        )
                if updates[target_inode] is None:
                    updates[target_inode] = log_prob
                else:
                    updates[target_inode] = np.log(np.exp(updates[target_inode]) + np.exp(log_prob))
        return updates

    def _collect_evidence_from_data(self, target, data, feature_states, collect_truths):
        evidence_sets = []
        truths = []
        for row in data:
            fs_indices = row.nonzero()[0] #np.argwhere(row == 1).flatten()
            # Collect evidence
            evidence = {}
            for fs_idx in fs_indices:
                feature, state = feature_states[fs_idx]
                if feature == target:
                    if collect_truths:
                        truths.append(state)
                    continue
                # Sanity check
                assert feature not in evidence
                evidence[feature] = state
            evidence_sets.append(evidence)
        return evidence_sets, truths

    def _get_prediction(self, update):
        # Remember that we are using log prob
        pred_state_prob = (-np.inf, None)
        for (_, state), log_prob in update.items():
            if log_prob is None:
                continue
            if log_prob > pred_state_prob[0]:
                pred_state_prob = (log_prob, state)
        return pred_state_prob[1]

    def predict(self, target:str, data, feature_states, collect_truths=False, heuristic=None, verbose=0, logger=None):
        if heuristic is None:
            reasoning_fn = self.update
        elif heuristic == 'fused_with_complete':
            reasoning_fn = self.heuristic_update_complete_fused
        else:
            raise ValueError('Unknown heuristic.')
        evidence_sets, truths = self._collect_evidence_from_data(target, data, feature_states, collect_truths)
        predictions = []
        i = 0
        for evidence in tqdm.tqdm(evidence_sets, desc='Reasoning on tests', leave=False, disable=verbose==0):
            update = reasoning_fn(target, evidence, verbose=verbose)
            pred = self._get_prediction(update)
            predictions.append(pred)
            i += 1
            if logger is not None:
                logger.report(i=i, total=len(evidence_sets))
        preds = np.array(predictions)
        if collect_truths:
            return preds, truths
        return preds


