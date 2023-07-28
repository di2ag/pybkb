import copy
import itertools
import numpy as np
import torch
from collections import defaultdict
from pomegranate.bayesian_network import BayesianNetwork

from pybkb.bn import BN
from pybkb.reason import BaseReasoner

class BNReasoner(BaseReasoner):
    def __init__(self, bn, data, feature_states):
        self.bn = bn
        self.data = data
        self.feature_states = feature_states
        self.rv_data = self.construct_rv_data(data, feature_states)
        self.pom_bn = self.to_pomegranate()
        super().__init__()

    def construct_rv_data(self, data, feature_states):
        rv_data = []
        for row in data:
            f_dict = {feature_states[i][0]: feature_states[i][1] for i in np.where(row == 1)[0]}
            new_row = []
            for feature in self.bn.rvs:
                new_row.append(self.bn.rv_state_indices_map[feature][f_dict[feature]])
            rv_data.append(np.array(new_row))
        return np.array(rv_data)

    def topological_sort_util(self, v, visited, stack):
        """ Modified from: https://www.geeksforgeeks.org/python-program-for-topological-sorting/
        """
        # Mark the current node as visited.
        visited[v] = True
        # Recur for all the vertices adjacent to this vertex
        if self.bn.pa[v] is not None:
            for i in self.bn.pa[v]:
                if visited[i] == False:
                    self.topological_sort_util(i, visited, stack)
        # Push current vertex to stack which stores result
        stack.append(v)
 
    def topological_sort(self):
        """ Modified from: https://www.geeksforgeeks.org/python-program-for-topological-sorting/
        """
        # Mark all the vertices as not visited
        visited = {rv: False for rv in self.bn.rvs}
        stack =[]
        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in self.bn.rvs:
            if visited[i] == False:
                self.topological_sort_util(i,visited,stack)
        return stack

    def to_pomegranate(self):
        """ Fuction to build the pomegranate structure of the BN based on our BN representation.
        """
        # Initialize the structure, will be in terms of rv indices
        structure = []
        for rv in self.bn.rvs:
            pa_set = self.bn.get_parents(rv)
            if not pa_set:
                structure.append(())
                continue
            structure.append(
                    tuple([self.bn.rv_indices_map[pa] for pa in pa_set])
                    )
        # Now build the pomegranate BN
        # At this point, no CPTs are learned.
        bn = BayesianNetwork(structure=structure)

        # Now learn/fit the parameters of this structure to the data.
        bn.fit(self.rv_data)
        return bn

    def update(self, target:str, evidence:dict):
        # Construct evidence array
        evidence_list = []
        for rv in self.bn.rvs:
            if rv not in evidence:
                evidence_list.append(-1)
                continue
            evidence_list.append(self.bn.rv_state_indices_map[rv][evidence[rv]])
        # Construct mask
        mask_list = [b>=0 for b in evidence_list]
        # Construct tensor versions
        evidence_tensor = torch.tensor([evidence_list])
        mask_tensor = torch.tensor([mask_list])
        # Construct masked tensor
        masked_update_vars = torch.masked.MaskedTensor(evidence_tensor, mask=mask_tensor)
        # Update
        updates = self.pom_bn.predict_proba(masked_update_vars)
        return updates[self.bn.rv_indices_map[target]]

    def _construct_pomegranate_data(self, target, data, feature_states, collect_truths):
        rv_data = self.construct_rv_data(data, feature_states)
        if collect_truths:
            truths = [row[self.bn.rv_indices_map[target]] for row in rv_data]
        else:
            truths = None
        # Construct masked tensors
        X = torch.tensor(rv_data)
        mask = [True for _ in range(rv_data.shape[1])]
        mask[self.bn.rv_indices_map[target]] = False
        mask = torch.tensor([mask])
        mask = mask.expand(rv_data.shape[0], -1)
        X_masked = torch.masked.MaskedTensor(X, mask=mask)
        return X_masked, truths

    def predict(self, target:str, data, feature_states, collect_truths=False):
        pom_data, truths = self._construct_pomegranate_data(target, data, feature_states, collect_truths)
        predictions = self.pom_bn.predict(pom_data)
        preds = np.array(predictions[:,self.bn.rv_indices_map[target]])
        if collect_truths:
            return preds, truths
        return preds
