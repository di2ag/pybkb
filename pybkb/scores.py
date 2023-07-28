from collections import defaultdict
import numpy as np
import tqdm

from pybkb.utils.probability import *

### Functions

def calculate_required_joint_probs(score_nodes, data, store, data_len, logger=None, verbose=True):
    # Extract unique joint probs
    joint_probs_to_calc = set()
    for score_node in tqdm.tqdm(score_nodes, desc='Collecting joints', leave=False, disable=not verbose):
        joint_probs_to_calc.update(score_node.extract_joint_probs())
    # Batch out the joint prob calcs
    #_, store = joint_probs_eff(data, parent_state_indices_list=list(joint_probs_to_calc), store=store, batch_size=2000)
    #indices_set = score_node.extract_joint_probs()
    for indices in tqdm.tqdm(joint_probs_to_calc, desc='Calculating joints', leave=False, disable=not verbose):
        #if indices not in joint_probs_to_calc:
        _, store = joint_prob_eff(data, data_len, parent_state_indices=indices, store=store)
    #joint_probs_to_calc.update(indices_set)
    return store

def calculate_scores_from_nodes(
        score_nodes,
        data,
        feature_states,
        store,
        feature_states_index_map=None,
        only=None,
        ):
    scores = defaultdict(dict)
    for score_node in score_nodes:
        if score_node.pa_set is None:
            pa_set = frozenset()
        else:
            pa_set = frozenset([str(idx) for idx in score_node.pa_set])
        score, store = score_node.calculate_score(
                data,
                feature_states,
                store,
                feature_states_index_map=feature_states_index_map,
                only=only,
                )
        scores[str(score_node.x)][pa_set] = score
    return scores, store

### Classes

class ScoreCollection:
    def __init__(
            self,
            node_encoding_len:float,
            score_node_obj,
            rv_level:bool=False,
            ):
        self.node_encoding_len = node_encoding_len
        self.rv_level = rv_level
        self.score_node_obj = score_node_obj
        self.scores = set()

    def optimize_memory(self):
        self.scores = np.array(list(self.scores))

    def split_collection(self, chunk_size=1e5):
        i = 0
        scores = []
        for score in self.scores:
            if i <= chunk_size:
                scores.append(score)
                i += 1
            if i == chunk_size:
                score_collection = ScoreCollection(self.node_encoding_len, self.score_node_obj, self.rv_level)
                score_collection.scores = scores
                yield score_collection
                scores = []
                i = 0
        if len(scores) > 0:
            score_collection = ScoreCollection(self.node_encoding_len, self.score_node_obj, self.rv_level)
            score_collection.scores = scores
            yield score_collection

    def add_score(self, x_pa:tuple):
        self.scores.add(x_pa)

    def add_scores(self, other_scores):
        self.scores.update(other_scores)

    def instantiate_node(self, x, pa):
        return self.score_node_obj(
                        x,
                        self.node_encoding_len,
                        rv_level=self.rv_level,
                        pa_set=pa,
                        )

    def instantiate_nodes(self, x_pa):
        x, pa = x_pa
        yield self.instantiate_node(x, pa)

    def instantiate_all_nodes(self, verbose=False):
        for x_pa in tqdm.tqdm(self.scores, desc='Instantiating Score Nodes', leave=False, disable=not verbose):
            yield instantiate_nodes(x_pa)

    def extract_joint_probs(self, verbose=False, position=0):
        for x_pa in tqdm.tqdm(self.scores, leave=False, disable=not verbose, desc='Extracting Required Joints', position=position):
            for node in self.instantiate_nodes(x_pa):
                for prob in node.extract_joint_probs():
                    yield prob

    def calculate_scores(self, data, data_eff, feature_states, store, feature_states_index_map, only, verbose, all_scores):
        scores = defaultdict(dict)
        for (x, pa) in tqdm.tqdm(self.scores, desc='Calculating Scores', leave=False, disable=not verbose):
            pa_set_formatted = frozenset([str(p) for p in pa])
            try:
                score = all_scores[(x, pa)]
            except KeyError:
                node = self.instantiate_node(x, pa)
                score, store = node.calculate_score(
                    data,
                    feature_states,
                    store,
                    feature_states_index_map=feature_states_index_map,
                    only=only,
                    data_eff=data_eff,
                    )
                all_scores[(x, pa)] = score
            scores[str(x)][pa_set_formatted] = score
        return dict(scores), store, all_scores

    def calculate_scores_vector(self, data, data_eff, feature_states, feature_states_index_map, store, only, logger):
        i = 0
        total = len(self.scores)
        scores = []
        logger.initialize_loop_reporting()
        for (x, pa) in self.scores:
            i += 1
            node = self.instantiate_node(x, pa)
            score, store = node.calculate_score(
                data,
                feature_states,
                store,
                feature_states_index_map=feature_states_index_map,
                data_eff=data_eff,
                only=only,
                )
            scores.append(score)
            logger.report(i=i, total=total)
        return np.array(scores)

    def match_all_scores(self, scores):
        if type(self.scores) == set:
            print('Warning: Scores is in set format which might not match desired scores sequence.')
        all_scores = dict()
        for x_pa, score in zip(self.scores, scores):
            all_scores[tuple(x_pa)] = score
        return all_scores

    def merge(self, other):
        if self.node_encoding_len != other.node_encoding_len:
            raise ValueError('Cannot merge collections with different encoding lengths.')
        if self.rv_level != other.rv_level:
            raise ValueError('Cannot merge collections that are at different RV levels.')
        if self.score_node_obj != other.score_node_obj:
            raise ValueError('Cannot merge collections with different scoring objects.')
        self.merge_scores(other.scores)

    def merge_scores(self, other_scores):
        self.scores.update(other_scores)

class ScoreNode:
    def __init__(
            self,
            x,
            node_encoding_len:float,
            rv_level:bool=False,
            pa_set:list=None,
            states:dict=None,
            indices:bool=True,
            )  -> None:
        """ Generic scoring node that captures a parent set and target variable and other generic
            information.

        Args:
            :param x: The target feature or feature state. 
            :type x: str or tuple
            :param node_encoding_len: Encoding length of a node in the model.
            :type node_encoding_len: float

        Kwargs:
            :param rv_level: Whether to score at the random variable level or the random variable instantiation level.
                Defaults to False.
            :type rv_level: bool
            :param pa_set: Parent set of x.
            :type pa_set: list
            :param states: Dictionary of every features set of states. Used in calculating scores on 
            the RV level, not necessary for instantiation level calculations.
            :type states: dict
            :param indices: Specifies if x and pa_set identifiers are indices in the data matrix. Defaults to True.
            :type indices: bool
        """
        if rv_level and indices:
            raise NotImplementedError('Must pass in feature and pa set NAMES not indices.')
        self.rv_level = rv_level
        self.x = x        
        if pa_set is None:
            self.pa_set = []
        else:
            self.pa_set = list(pa_set)
        self.node_encoding_len = node_encoding_len
        self.indices = indices
        # Reduces state dictionary to only what is in the node if rv level node calculator
        if states is not None and self.rv_level:
            self.states = {feature: states[feature] for feature in self.pa_set + [x]}
        else:
            self.states = None

    def _calc_instantiated_score(self, data, feature_states_index_map, store):
        pass
    
    def _calc_rvlevel_score(self, data, feature_states, feature_states_map):
        pass

    def _extract_joint_probs(self):
        pass

    def extract_joint_probs(self):
        return self._extract_joint_probs()

    def calculate_score(
            self,
            data:np.array,
            feature_states:list,
            store:dict,
            feature_states_index_map:dict=None,
            feature_states_map:dict=None,
            only:str=None,
            data_eff:dict=None,
            ):
        """ Generic method that is overwritten to calculate score.
            
        Args:
            :param data: Full database in a binary matrix.
            :type data: np.array
            :param feature_states: A list of all feature states exactly as it appears in the data matrix.
            :type feature_states: list
            :param store: A store database of calculated joint probabilties.
            :type store: dict

        Kwargs:
            :param feature_states_index_map: A dictionary mapping feature state tuples to appropriate column index in the data matrix.
            :type feature_states_index_map: dict
            :param feature_states_map: A dictionary keyed by feature with values of as the list of available states the feature can take. 
                Use the build_feature_state_map function in pybkb.utils.probability to get the correct format.
            :type feature_states_map: dict
            :param only: Return only the data score or model score or both. Options: data, model, both, None. Defaults to None which means both.
            :type only: str
        """
        if not self.rv_level:
            # Calculate structure MDL
            struct_mdl = (len(self.pa_set) + 1)*self.node_encoding_len
            # Calculate instantiated data MDL
            data_mdl, store = self._calc_instantiated_score(data, feature_states_index_map, store, data_eff)
            # Note: Number of atomic events represented by an S-node is just 1
            #if data_mdl == 0:
            #    data_mdl = 1e-10
        else:
            # Not returning a store
            store = None
            # Calculate node structure MDL
            struct_mdl = len(self.pa_set) * self.node_encoding_len
            struct_mdl += (len(self.states[self.x]) - 1) * np.prod([len(self.states[pa]) for pa in self.pa_set])
            num_atomic_events = len(self.states[self.x]) * np.prod([len(self.states[pa]) for pa in self.pa_set])
            # Calculate random variable level data MDL
            data_mdl = self._calc_rvlevel_score(data, feature_states, feature_states_map, store, data_eff)
            #TODO: Think about this
            #data_mdl *= num_atomic_events
        if only is None:
            return -data_mdl - struct_mdl, store
        if only == 'data':
            return -data_mdl, store
        if only == 'model':
            return -struct_mdl, store
        if only == 'both':
            return -data_mdl, -struct_mdl, store
        raise ValueError(f'Unknown option {only}. Must be one of [data, model, both].')

    def __hash__(self):
        return hash((self.x, frozenset(self.pa_set)))


class MdlEntScoreNode(ScoreNode):
    def __init__(
            self,
            x,
            node_encoding_len:float,
            rv_level:bool=False,
            pa_set:list=None,
            states:dict=None,
            indices:bool=True,
            )  -> None:
        """ MDL scoring node using conditional entropy and captures a parent set and target variable and other generic
            information.

        Args:
            :param x: The target feature or feature state. 
            :type x: str or tuple
            :param node_encoding_len: Encoding length of a node in the model.
            :type node_encoding_len: float

        Kwargs:
            :param rv_level: Whether to score at the random variable level or the random variable instantiation level.
                Defaults to False.
            :param pa_set: Parent set of x.
            :type pa_set: list
            :param states: Dictionary of every features set of states. Used in calculating scores on 
            the RV level, not necessary for instantiation level calculations.
            :type states: dict
            :param indices: Specifies if x and pa_set identifiers are indices in the data matrix. Defaults to True.
            :type indices: bool
        """
        super().__init__(x, node_encoding_len, rv_level, pa_set, states, indices)

    def _calc_instantiated_score(self, data, feature_states_index_map, store, data_eff):
        """ Calculate MDL with conditional entropy weight on the random variable instantiation level.
        """
        # Get x and pa indices in data set
        if self.indices:
            x_state_idx = self.x
            parent_state_indices = self.pa_set
        else:
            x_state_idx = feature_states_index_map[self.x]
            parent_state_indices = [feature_states_index_map[pa] for pa in self.pa_set]
        # Calculate data MDL
        return instantiated_conditional_entropy(
                data, 
                x_state_idx, 
                parent_state_indices, 
                store=store, 
                data_eff=data_eff
                )

    def _calc_rvlevel_score(self, data, feature_states, feature_states_map, store, data_eff):
        """ Calculate MDL with conditional entropy weight on the random variable level.
        """
        # If no parents use variable entropy as data_mdl
        h, _ = conditional_entropy(
                data,
                feature_states,
                self.x, 
                self.pa_set, 
                store=store, 
                data_eff=data_eff, 
                feature_states_map=feature_states_map
                )
        return h

    def _extract_joint_probs(self):
        if self.pa_set is None:
            pa_set = []
        else:
            pa_set = self.pa_set
        joints = set([frozenset([self.x] + pa_set)])
        joints.add(frozenset(pa_set))
        return joints


class MdlMutInfoScoreNode(ScoreNode):
    def __init__(
            self,
            x,
            node_encoding_len:float,
            rv_level:bool=False,
            pa_set:list=None,
            states:dict=None,
            indices:bool=True,
            )  -> None:
        """ MDL scoring node using mutual information and captures a parent set and target variable and other generic
            information.

        Args:
            :param x: The target feature or feature state. 
            :type x: str or tuple
            :param node_encoding_len: Encoding length of a node in the model.
            :type node_encoding_len: float

        Kwargs:
            :param rv_level: Whether to score at the random variable level or the random variable instantiation level.
                Defaults to False.
            :param pa_set: Parent set of x.
            :type pa_set: list
            :param states: Dictionary of every features set of states. Used in calculating scores on 
            the RV level, not necessary for instantiation level calculations.
            :type states: dict
            :param indices: Specifies if x and pa_set identifiers are indices in the data matrix. Defaults to True.
            :type indices: bool
        """
        super().__init__(x, node_encoding_len, rv_level, pa_set, states, indices)

    def _calc_instantiated_score(self, data, feature_states_index_map, store, data_eff):
        """ Calculate MDL with mutual information weight on the random variable instantiation level.
        """
        # Get x and pa indices in data set
        if self.indices:
            x_state_idx = self.x
            parent_state_indices = self.pa_set
        else:
            x_state_idx = feature_states_index_map[self.x]
            parent_state_indices = [feature_states_index_map[pa] for pa in self.pa_set]
        # Calculate data MDL
        return instantiated_mutual_info(
                data,
                x_state_idx,
                parent_state_indices,
                store,
                data_eff=data_eff
                )

    def _calc_rvlevel_score(self, data, feature_states, feature_states_map, store, data_eff):
        """ Calculate MDL with mutual information weight on the random variable level.
        """
        # Cacluate Mutual Information weight
        mi, _ = mutual_info(
                data, 
                feature_states,
                self.x, 
                self.pa_set,
                store=store,
                data_eff=data_eff,
                feature_states_map=feature_states_map,
                )
        return mi
    
    def _extract_joint_probs(self):
        if self.pa_set is None:
            pa_set = []
        else:
            pa_set = self.pa_set
        joints = set([frozenset([self.x] + pa_set)])
        joints.add(frozenset(pa_set))
        joints.add(frozenset([self.x]))
        return joints
