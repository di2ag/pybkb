import os
import numpy as np
import itertools
import logging
import contextlib
import time
import tqdm
from operator import itemgetter
from pygobnilp.gobnilp import Gobnilp
from gurobipy import GRB

from pybkb.utils.probability import *
from pybkb.utils.mp import MPLogger
from pybkb.scores import *
from pybkb.learn.report import LearningReport
from pybkb.bn import BN


class BNGobnilpBackend:
    def __init__(
            self,
            score:str,
            palim:int=None,
            only:str=None,
            ) -> None:
        """ BKB Gobnilp DAG learning backend.

        Args:
            :param score: Name of scoring function. Choices: mdl_mi, mdl_ent.
            :type score: str

        Kwargs:
            :param palim: Limit on the number of parent sets. 
            :type palim: int
            :param only: Return only the data score or model score or both. Options: data, model, None. Defaults to None which means both.
            :type only: str
        """
        if score == 'mdl_mi':
            self.score_node = MdlMutInfoScoreNode
        elif score == 'mdl_ent':
            self.score_node = MdlEntScoreNode
        else:
            raise ValueError(f'Unknown score: {score}')
        self.palim = palim
        self.score = score
        self.only = only 
        self.store = {}

    def calculate_all_local_scores(
            self,
            data:np.array,
            features:set,
            states:dict,
            feature_states_map:dict,
            feature_states:list,
            filepath:str=None,
            verbose:bool=False,
            logger=None,
            reset:bool=True,
            all_scores:dict=None
            ) -> dict:
        """ Generates local scores for Gobnilp optimization
        
        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param features: A set of feature names.
            :type features: set
            :param states: A dictionary of states for each feature. Differs from feature_states_map
                because it doesn't contain the index of in the feature_states.
            :type states: dict
            :param feature_states_map: A dictionary keyed by feature name with 
                values equalling a list of allowed states of the form [(idx, state_name), ...].
                Use the pybkb.utils.probability.build_feature_state_map function to build this map.
            :type feature_states_map: dict
            :param feature_states: List of feature instantiations.
            :type feature_states: list

        Kwargs:
            :param filepath: Optional filepath where local scores will be written. Defaults None.
            :type filepath: str
        """
        node_encoding_len = np.log2(len(features))
        # Setup parent set limit if None
        if self.palim is None:
            palim = len(features) - 1
        else:
            palim = self.palim
        if reset:
            # Reset store
            self.store = build_probability_store()
            # Calculate scores
            scores = defaultdict(dict)
        # Initialize scores
        if all_scores is None:
            scores = defaultdict(dict)
        else:
            scores = all_scores
        j = 0
        # Calculate MDL scores
        for feature in tqdm.tqdm(features, desc='Scoring', disable=not verbose, leave=False):
            for i in range(palim + 1):
                if i == 0:
                    node_hash = (feature, frozenset())
                    if node_hash in scores:
                        continue
                    node = self.score_node(
                            feature,
                            node_encoding_len,
                            states=states,
                            indices=False,
                            rv_level=True,
                            )
                    score, self.store = node.calculate_score(
                            data,
                            feature_states,
                            self.store,
                            feature_states_map=feature_states_map,
                            only=self.only,
                            )
                    scores[feature][frozenset()] = score
                    continue
                for pa_set in itertools.combinations(set.difference(features, {feature}), r=i):
                    node_hash = (feature, frozenset([str(pa) for pa in pa_set])) 
                    if node_hash in scores:
                        continue
                    node = self.score_node(
                            feature,
                            node_encoding_len,
                            pa_set=pa_set,
                            states=states,
                            indices=False,
                            rv_level=True,
                            )
                    score, self.store = node.calculate_score(
                            data,
                            feature_states,
                            self.store,
                            feature_states_map=feature_states_map,
                            only=self.only,
                            )
                    scores[feature][frozenset(pa_set)] = score
            j += 1
            if logger is not None:
                logger.report(i=j, total=len(features))
        if filepath:
            # Make into string format
            s = f'{len(features)}\n'
            for feature, pa_scores in scores.items():
                s += f'{feature} {len(pa_scores)}\n'
                for pa_set, score in pa_scores.items():
                    if pa_set is None:
                        pa_set = []
                    s += f'{score} {len(pa_set)}'
                    for pa in pa_set:
                        s += f' {pa}'
                    s += '\n'
            # Write to file
            with open(filepath, 'w') as f_:
                f_.write(s)
        return dict(scores)

    def learn(
            self,
            data:np.array,
            feature_states:list,
            verbose:bool=False,
            scores:dict=None,
            store:dict=None,
            begin_stage:str=None,
            end_stage:str=None,
            nsols:int=1,
            kbest:bool=True,
            mec:bool=False,
            pruning:bool=False,
            num_workers:int=None,
            scores_filepath:str=None,
            ):
        """ Learns the best BN from the data.

        Args:
            :param data: Full database to learn over.
            :type data: np.array
            :param feature_states: List of feature instantiations.
            :type feature_states: list
            :param srcs: A list of source names for each data row.
            :type srcs: list

        Kwargs:
            :param verbose: Whether to print progress bars.
            :type verbose: bool
            :param scores: Pre-calculated scores dictionary.
            :type scores: dict
            :param store: Pre-calculated store of joint probabilities.
            :type store: dict
            :param begin_stage: Stage to begin learning. Passing 'scores' will assume a scores file has been passed and
                will proceed directly to BKF learning.
            :type begin_stage: str
            :param end_stage: Stage to end learning. Passing 'scores' will end learning after all
                scores are calculated.
            :type end_stage: str
            :param nsols: Number of BKBs to learn per example.
            :type nsols: int
            :param kbest: Whether the nsols learned BKBs should be a highest scoring set of nsols BKBs.
            :type kbest: bool
            :param mec: Whether only one BKB per Markov equivalence class should be feasible.
            :type mec: bool
            :param pruning: Whether not to include parent sets which cannot be optimal when acyclicity is the only constraint.
            :type pruning: bool
            :param num_workers: Number of workers in a distributed pool to calculate scores and joint probabilities.
            :type num_workers: int
            :param scores_filepath: Optional filepath where local scores will be written. Defaults None.
            :type scores_filepath: str
        """
        # Initialize report
        report = LearningReport('gobnilp', True)
        # Collect features and states
        features = []
        states = defaultdict(list)
        feature_states_map = build_feature_state_map(feature_states)
        for f, s in feature_states:
            features.append(f)
            states[f].append(s)
        features = set(features)
        states = dict(states)
        # Reset store unless passed
        if store is None:
            self.store = build_probability_store()
        else:
            self.store = store
        # Calculate local scores unless passed
        if scores is None:
            scores = self.calculate_all_local_scores(data, features, states, feature_states_map, feature_states, verbose=verbose)
        if end_stage == 'scores':
            return scores, self.store
        elif end_stage is None:
            pass
        else:
            raise ValueError(f'Unknown end stage option: {end_stage}.')
        # Update report with calls to joint calculator
        #TODO: Currently not returning a store because of new entropy calc for RV level.
        #report.update_from_store(self.store)
        
        f = open(os.devnull, 'w')
        with contextlib.redirect_stdout(f):
            m = Gobnilp()
            # Start the learning but stop before learning to add constraints
            m.learn(local_scores_source=scores, end='MIP model', nsols=nsols, kbest=kbest, pruning=pruning, mec=mec)
            # Grab all the adjacency variables
            adj = [v for p, v in m.adjacency.items()]
            # Add a constraint that at the DAG must be connected (need to subtract one to make a least a tree)
            m.addLConstr(sum(adj), GRB.GREATER_EQUAL, len(features) - 1)
        # Close devnull file as to not get resource warning
        f.close()
        # Learn the DAG
        report.start_timer()
        m.learn(local_scores_source=scores, start='MIP model', gurobi_output=False, verbose=0)
        # Add learning time to report
        #learn_time = time.time() - start_time
        report.add_bn_learn_time(report.end_timer())
        # Convert learned_bn from pygobnilp to interal bn representation so we can score like a BKB
        bn = BN.from_bnlearn_modelstr(m.learned_bn.bnlearn_modelstring(), states)
        data_score, model_score = bn.score_like_bkb(
                data,
                feature_states,
                self.score,
                feature_states_map,
                only='both',
                store=self.store,
                )
        return bn, m, report
        ''' 
        # Learn the best DAG from these local scores using Gobnilp
        m = Gobnilp()
        # Start the learning but stop before learning to add constraints
        m.learn(local_scores_source=scores, end='MIP model')
        # Grab all the adjacency variables
        adj = [v for p, v in m.adjacency.items()]
        # Add a constraint that at the DAG must be connected
        m.addLConstr(sum(adj), GRB.GREATER_EQUAL, len(features) - 1)
        # Learn the DAG
        report.start_timer()
        m.learn(local_scores_source=scores, start='MIP model')
        '''

    def _make_snode_hash(self, snode_dict):
        head_hash = make_hash(snode_dict["Head"])
        prob_hash = hash(snode_dict["Probability"])
        tail_hash = make_hash(snode_dict["Tail"])
        print(head_hash)
        print(prob_hash)
        print(tail_hash)
        return hash((head_hash, prob_hash, tail_hash))
        report.add_bn_like_bkb_scores(data_score, model_score)
        # Finialize report
        report.finalize()
        return m.learned_bn, m, report
