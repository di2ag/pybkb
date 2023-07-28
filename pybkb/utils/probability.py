import itertools
import tqdm
import numpy as np
import pandas as pd
import numba as nb
from numba import njit
from collections import defaultdict

from pybkb.exceptions import InvalidProbabilityError

def build_probability_store():
    """ Helper function that will build the probability store so that we can track
    number of calls to the joint probability, and other potential metrics.
    """
    store = {
            "__ncalls__": 0,
            "__nhashlookups__": 0,
            }
    return store

def _joint_prob(data, x_state_idx=None, parent_state_indices=None, store=None):
    if store is None:
        store = build_probability_store()
    if x_state_idx is None:
        x_state_idx = []
    else:
        x_state_idx = [x_state_idx]
    if parent_state_indices is None:
        parent_state_indices = []
    indices = frozenset(x_state_idx + parent_state_indices)
    if indices in store:
        prob = store[indices]
        store['__nhashlookups__'] += 1
    else:
        count = 0
        for row in data:
            truth = True
            for s_idx in indices:
                if row[s_idx] != 1:
                    truth = False
                    break
            if truth:
                count += 1
        prob = count / len(data)
        store[indices] = prob
    store['__ncalls__'] += 1
    return prob, store

def get_max_joint_sets(row, joint_len):
    """ Returns the maximum joint instantiation set or sets based on required joint length
    or the number of feature states present in the data row.
    
    Args:
        :param row: A one-hot encoded data row from the dataset under study.
        :type row: np.array
        :param joint_len: The length of the required joint instantiation.
        :type palim: int
    """
    max_joints = set()
    # Extract indices present in data row
    indices = np.where(row == 1)[0]
    for combo in itertools.combinations(indices, r=min([joint_len, len(indices)])):
        max_joints.add(frozenset(list(combo)))
    return max_joints

def expand_max_joint_set(joint_set):
    """ Expands the joint set into small joint sets up to the length of the passed
    joint set.

    Args:
        :param joint_set: The joint set to be expanded to all possible smaller subsets.
        :type joint_set: frozenset
    """
    joints = {joint_set}
    for r in range(1, len(joint_set)):
        for sub_joint in itertools.combinations(joint_set, r=r):
            joints.add(sub_joint)
    return joints

@njit
def get_test_arr(indices, feature_states_len):
    test_arr = np.zeros(feature_states_len)
    test_arr[indices] = 1
    return test_arr

@njit
def _calc_joint_prob(data, test_arr):
    count_arr = np.dot(data, test_arr) 
    count = len(count_arr[count_arr == np.count_nonzero(test_arr)])
    prob = count / data.shape[0]
    return prob

@njit
def calc_joint_prob(data, indices):
    test_arr = get_test_arr(indices, data.shape[1])
    return _calc_joint_prob(data, test_arr)

@njit
def calc_joint_probs(data, test_arr):
    count_arr = np.dot(data, test_arr)
    maxes = np.count_nonzero(test_arr, axis=0)
    probs = (data.shape[0] - np.count_nonzero(count_arr - maxes, axis=0)) / data.shape[0]
    return probs

@njit
def fast_intersect(l1, l2):
    l3 = np.array([i for i in l1 for j in l2 if i==j])
    return np.unique(l3)

@njit
def calc_joint_prob_eff(data, indices, data_len):
    l1 = data[indices[0]]
    for i in range(1, len(indices)):
        l1 = fast_intersect(l1, data[indices[i]])
    count = len(l1)
    # Doesn't work with numba but keeping in as a reminder
    #count = len(set.intersection(*[set(data[idx]) for idx in indices]))
    prob = count / data_len
    return prob

def joint_prob_eff(data, data_len, x_state_idx=None, parent_state_indices=None, store=None):
    if store is None:
        store = build_probability_store()
    if x_state_idx is None:
        x_state_idx = []
    else:
        x_state_idx = [x_state_idx]
    if parent_state_indices is None:
        parent_state_indices = []
    else:
        parent_state_indices = list(parent_state_indices)
    indices = frozenset(x_state_idx + parent_state_indices)
    if indices in store:
        prob = store[indices]
        store['__nhashlookups__'] += 1
    elif len(indices) == 0:
        prob = 1
        store[indices] = 1
    else:
        prob = calc_joint_prob_eff(data, nb.typed.List(indices), data_len)
        store[indices] = prob
    store['__ncalls__'] += 1
    return prob, store

def joint_probs_eff(data, data_len, indices_list, verbose=False, position=0, store=None):
    if store is None:
        store = build_probability_store()
    probs = []
    with tqdm.tqdm(total=len(indices_list), desc='Calculating Probabilities', leave=False, position=position, disable=not verbose) as pbar:
        while indices_list:
            indices = indices_list.pop()
            #print(indices)
            prob, store = joint_prob_eff(data, data_len, parent_state_indices=indices, store=store)
            del indices
            probs.append(prob)
    #print(probs)
    return np.array(probs), store

def joint_probs_eff_from_sc(data, data_len, score_collection, verbose=False, position=0, store=None):
    if store is None:
        store = build_probability_store()
    for indices in score_collection.extract_joint_probs(verbose, position):
        _, store = joint_prob_eff(data, data_len, parent_state_indices=indices, store=store)
    return store

def joint_prob(data, x_state_idx=None, parent_state_indices=None, store=None):
    if store is None:
        store = build_probability_store()
    if x_state_idx is None:
        x_state_idx = []
    else:
        x_state_idx = [x_state_idx]
    if parent_state_indices is None:
        parent_state_indices = []
    else:
        parent_state_indices = list(parent_state_indices)
    indices = frozenset(x_state_idx + parent_state_indices)
    if indices in store:
        prob = store[indices]
        store['__nhashlookups__'] += 1
    elif len(indices) == 0:
        prob = 1
        store[indices] = 1
    else:
        prob = calc_joint_prob(data, np.array(list(indices)))
        store[indices] = prob
    store['__ncalls__'] += 1
    return prob, store

"""
def _joint_probs(data, x_state_idx_list=None, parent_state_indices_list=None, store=None, logger=None):
    if store is None:
        store = build_probability_store()
    # Process indices_list
    if x_state_idx_list is None:
        indices_list = [frozenset(parent_state_indices) for parent_state_indices in parent_state_indices_list if parent_state_indices is not None]
    elif parent_state_indices_list is None:
        indices_list = [frozenset([x_state_idx]) for x_state_idx in x_state_idx_list if x_state_idx is not None]
    else:
        indices_list = []
        for x_state_idx, parent_state_indices in zip(x_state_idx_list, parent_state_indices_list):
            if x_state_idx is not None and parent_state_indices is not None:
                indices_list.append(frozenset([x_state_idx] + parent_state_indices))
            elif x_state_idx is None:
                indices_list.append(frozenset(parent_state_indices))
            elif parent_state_indices is None:
                indices_list.append(frozenset([x_state_idx]))
        indices_list = list(set(indices_list))
    probs = np.zeros(len(indices_list))
    indices_list_to_check = []
    for idx, indices in enumerate(indices_list):
        if indices in store:
            prob = store[indices]
            store['__nhashlookups__'] += 1
            probs[idx] = prob
            continue
        indices_list_to_check.append((idx, indices))
    # Now check the indices we haven't found in the store
    counts = defaultdict(int)
    total = len(data)
    for i, row in enumerate(data):
        for _, indices in indices_list_to_check:
            truth = True
            for s_idx in indices:
                if row[s_idx] != 1:
                    truth = False
                    break
            if truth:
                counts[indices] += 1
        if logger is not None:
            logger.report(i=i, total=total)
    # Now calculate probs
    for idx, indices in indices_list_to_check:
        prob = counts[indices] / len(data)
        store[indices] = prob
        store['__ncalls__'] += 1
        probs[idx] = prob
    return probs, store
"""
"""
def joint_probs(data, x_state_idx_list=None, parent_state_indices_list=None, store=None, logger=None, batch_size=1000, verbose=True):
    if store is None:
        store = build_probability_store()
    # Process indices_list
    #print('Preprocessing lists')
    if x_state_idx_list is None:
        indices_list = [frozenset(parent_state_indices) for parent_state_indices in parent_state_indices_list if parent_state_indices is not None]
    elif parent_state_indices_list is None:
        indices_list = [frozenset([x_state_idx]) for x_state_idx in x_state_idx_list if x_state_idx is not None]
    else:
        indices_list = []
        for x_state_idx, parent_state_indices in zip(x_state_idx_list, parent_state_indices_list):
            if x_state_idx is not None and parent_state_indices is not None:
                indices_list.append(frozenset([x_state_idx] + parent_state_indices))
            elif x_state_idx is None:
                indices_list.append(frozenset(parent_state_indices))
            elif parent_state_indices is None:
                indices_list.append(frozenset([x_state_idx]))
    #print('Getting lists to check')
    probs = np.zeros(len(indices_list))
    test_arr = []
    indices_list_to_check_orgin_indices = []
    calc_probs = []
    num_batches = len(indices_list) // batch_size + 1
    with tqdm.tqdm(desc='Processing Prob Batches', leave=False, disable=not verbose, total=num_batches) as pbar:
        for idx, indices in enumerate(indices_list):
            if indices in store:
                prob = store[indices]
                store['__nhashlookups__'] += 1
                probs[idx] = prob
                continue
            if len(indices) == 0:
                probs[idx] = 1
                store[indices] = 1
                continue
            test_arr.append(get_test_arr(np.array(list(indices)), data.shape[1]))
            indices_list_to_check_orgin_indices.append(idx)
            if len(test_arr) == batch_size:
                _test_arr = np.array(test_arr).transpose()
                _calc_probs = calc_joint_probs(data, _test_arr)
                calc_probs.append(_calc_probs)
                pbar.update(1)
                test_arr = []
        # Process remainder
        _test_arr = np.array(test_arr).transpose()
        _calc_probs = calc_joint_probs(data, _test_arr)
        calc_probs.append(_calc_probs)
        pbar.update(1)
    calc_probs = np.concatenate(calc_probs)
    #print('Putting things back in order.')
    for i, prob_idx in enumerate(indices_list_to_check_orgin_indices):
        probs[prob_idx] = calc_probs[i]
        indices = indices_list[prob_idx]
        store[indices] = calc_probs[i]
        store['__ncalls__'] += 1
    '''
    print('Forming test array')
    test_arr = np.zeros((data.shape[1], len(indices_list_to_check)))
    maxes = []
    for j, (_, indices) in enumerate(indices_list_to_check):
        maxes.append(len(indices))
        for i in indices:
            test_arr[i,j] = 1
    print('Multiplying matrices')
    count_arr = np.matmul(data, test_arr)
    maxes = np.array(maxes)
    print('Calculating probs.')
    calc_probs = (data.shape[0] - np.count_nonzero(count_arr - maxes, axis=0)) / data.shape[0]
    # Add calculated probs into store and results
    '''
    return probs, store
"""

def conditional_prob(
        data:np.array,
        x_state_idx:int,
        parent_state_indices:list,
        store=None,
        data_eff=None,
        ):
    """ Calculates the conditional probability p(X=x | \pi(X=x)).

    Args:
        :param data: The standard one-hot encoded BKB data.
        :type data: np.array
        :param x_state_idx: The r.v. instantiation X=x index as governed by the data sets feature states list.
        :type x_state_idx: int
        :param parent_state_indices_list: A list containing a list of parent r.v. instantiations for every parent r.v. 
            according to the data set feature states list. Example: [[pa_1_1, pa_1_2, pa_1_3], ..., [pa_n_1, pa_n_2, ..., pa_n_m]]
        :type parent_state_indices_list: list

    Kwargs:
        :param store: The probability store which can and should be updated to save time.
        :type store: dict
        :param data_eff: The bkb data formatted for efficiency as given by
            pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
        :type data_eff: list
    """
    if data_eff is not None:
        p_xp, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx, parent_state_indices, store)
        p_p, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    else:
        p_xp, store = joint_prob(data, x_state_idx, parent_state_indices, store)
        p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    if p_xp == 0:
        return 0, store
    try:
        prob = p_xp / p_p
    except ZeroDivisionError:
        if p_xp > 0:
            raise InvalidProbabilityError(p_xp, p_p=p_p)
        return 0
    return prob, store

def extract_joint_prob_vector(
        data:np.array,
        x_state_indices:list,
        parent_state_indices:list,
        store=None,
        data_eff=None,
        ):
    """ Extracts the joint probability vector from a r.v. instantation X=x and an 
    associated parent set instantiations \pi(X)=\pi(x)_i.

    Args:
        :param data: The standard one-hot encoded BKB data.
        :type data: np.array
        :param x_state_idx: The r.v. instantiation X=x index as governed by the data sets feature states list.
        :type x_state_idx: int
        :param parent_state_indices_list: A list containing a list of parent r.v. instantiations for every parent r.v. 
            according to the data set feature states list. Example: [[pa_1_1, pa_1_2, pa_1_3], ..., [pa_n_1, pa_n_2, ..., pa_n_m]]
        :type parent_state_indices_list: list

    Kwargs:
        :param store: The probability store which can and should be updated to save time.
        :type store: dict
        :param data_eff: The bkb data formatted for efficiency as given by
            pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
        :type data_eff: list
    """
    P_Xp = []
    for x_state_idx in x_state_indices:
        if data_eff is not None:
            p_xp, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx, list(parent_state_indices), store)
        else:
            p_xp, store = joint_prob(data, x_state_idx, list(parent_state_indices), store)
        P_Xp.append(p_xp)
    return np.array(P_Xp)

def extract_joint_prob_matrix(
        data:np.array,
        x_state_indices:list,
        parent_state_indices_list:list,
        store=None,
        data_eff=None,
        ):
    """ Extracts the joint probability matrix for a r.v. X and an associated parent set \pi(X).

    Args:
        :param data: The standard one-hot encoded BKB data.
        :type data: np.array
        :param x_state_indices: The r.v. instantiations, X=x_i, indices for a given r.v. as governed by the data sets feature states list.
        :type x_state_indices: list
        :param parent_state_indices_list: A list containing a list of parent r.v. instantiations for every parent r.v. 
            according to the data set feature states list. Example: [[pa_1_1, pa_1_2, pa_1_3], ..., [pa_n_1, pa_n_2, ..., pa_n_m]].append 
        :type parent_state_indices_list: list

    Kwargs:
        :param store: The probability store which can and should be updated to save time.
        :type store: dict
        :param data_eff: The bkb data formatted for efficiency as given by
            pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
        :type data_eff: list
    """
    joint_prob_matrix = []
    for parent_state_indices in itertools.product(*parent_state_indices_list):
        joint_prob_matrix.append(
                extract_joint_prob_vector(
                    data,
                    x_state_indices,
                    parent_state_indices,
                    store=store,
                    data_eff=data_eff,
                    )
                )
    return np.array(joint_prob_matrix)

def conditional_instantiated_entropy_from_probs(
        p_xp:float,
        p_p:float=None,
        p_xgp:float=None
        ):
    """ Function calculates the conditional instantiated entropy H(x|\pi(x)), where x is a random 
    variable instantiation and \pi(x) is a parent set instantiation, i.e. the parent r.v. 
    instantiations that this instantiation of x is conditioned.

    Reference Equation:
        H(x|\pi(x)) = - p(x, \pa(x)) \times \log_2[ p(x | \pi(x)) ]
    
    Args:
        :param p_xp: The joint probability value p(X=x, \pi(X)=\pi(x)).
        :type p_xp: float
    
    Kwargs:
        :param p_p: The joint probability value p(\pi(X) = \pi(x)).
        :type p_p: float
        :param p_xgp: The joint probability value p(x | \pi(x)).
        :type p_xgp: float
    """
    if p_p is None and p_xgp is None:
        raise ValueError('p_p and p_xgp can not both be None')
    if p_xgp is None:
        try:
            p_xgp = p_xp / p_p
        except ZeroDivisionError:
            if p_xp > 0:
                raise InvalidProbabilityError(p_xp, p_p=p_p)
            return 0
        except TypeError:
            raise ValueError('p_p can not be None when p_xgp is None.')
    if p_xgp == 0:
        return 0
    return - p_xp * np.log2(p_xgp)

def conditional_entropy_from_probs(joint_prob_matrix:np.array):
    """ Function that takes a joint probability matrix (X, \pi(X)) ~ p and calculates the
    r.v. conditional entropy. 
    
    Reference Equation:
        H(X|\pi(X)) = - \sum_{(x, \pi(x) \in X \times \pi(X)} p(x, \pa(x)) \times \log_2[ p(x | \pi(x)) ]
    
    Args
        :param joint_prob_matrix: The joint distribution matrix where the cell value for any i-row
            j-column represents the probability of \pa(x)_i, x_j, i.e., p(\pa(x)_i, x_j).
        :type joint_prob_matrix: np.array

    """
    with np.errstate(invalid='ignore'):
        P_XgP = np.nan_to_num(joint_prob_matrix / np.sum(joint_prob_matrix, axis=1)[:,None])
    entropy = 0
    for p_xp, p_xgp in zip(joint_prob_matrix.ravel(), P_XgP.ravel()):
        entropy += conditional_instantiated_entropy_from_probs(p_xp, p_xgp=p_xgp)
    return entropy
    
def instantiated_conditional_entropy(
        data:np.array,
        x_state_idx:int,
        parent_state_indices:list,
        store:dict=None,
        data_eff:list=None
        ):
    """ Function to calculate instantiated conditional entropy from data given feature state
    indices.

    Args:
        :param data: The standard one-hot encoded BKB data.
        :type data: np.array
        :param x_state_idx: The r.v. instantiation X=x index as governed by the data sets feature states list and  
            which is conditioned upon the parent set instantiations.
        :type x_state_idx: int
        :param parent_state_indices: The r.v. instantiations in the set of parent set, \pi(x), and are also indices
            according to the data set feature states list. 
        :type parent_state_indices: list

    Kwargs:
        :param store: The probability store which can and should be updated to save time.
        :type store: dict
        :param data_eff: The bkb data formatted for efficiency as given by
            pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
        :type data_eff: list
    """
    if data_eff is not None:
        p_xp, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx, parent_state_indices, store)
        p_p, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    else:
        p_xp, store = joint_prob(data, x_state_idx, parent_state_indices, store)
        p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    if p_xp == 0:
        return 0, store
    return conditional_instantiated_entropy_from_probs(p_xp, p_p=p_p), store

def conditional_entropy(
        data:np.array,
        feature_states:list,
        X,
        pa_set,
        store:dict=None,
        data_eff:list=None,
        feature_states_map=None,
        ):
    """ Function to calculate conditional entropy from bkb data given a feature X and a set of parent
    features pa_X = \pi(X).

    Args:
        :param data: The standard one-hot encoded BKB data.
        :type data: np.array
        :param feature_states: The standard feature_states list for the encoded BKB data.
        :type feature_states: list
        :param X: The random variable that will be continued on, must match a feature in the feature_states list.
        :type X: str,int
        :param pa_set: A set of parent r.v.'s, all parent r.v.'s must match a feature in the feature_states list.
        :type pa_set: list

    Kwargs:
        :param store: The probability store which can and should be updated to save time.
        :type store: dict
        :param data_eff: The bkb data formatted for efficiency as given by
            pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
        :type data_eff: list
        :param feature_states_map: A dictionary mapping features (r.v's) to all their respective instantiation 
            indices as given by the function pybkb.utils.probability.build_feature_state_map(no_state_names=True).  Will be built on demand if not passed.
        :type feature_states_map: dict
    """
    if not feature_states_map:
        feature_states_map = build_feature_state_map(feature_states)
    # Construct joint probability matrix
    joint_prob_matrix = extract_joint_prob_matrix(
            data,
            feature_states_map[X],
            [feature_states_map[pa] for pa in pa_set],
            store=store,
            data_eff=data_eff,
            )
    # Calculate conditional entropy
    return conditional_entropy_from_probs(joint_prob_matrix), feature_states_map

def instantiated_mutual_info_from_probs(
        p_x:float,
        p_xp:float,
        p_p:float,
        ):
    """ Function calculates the instantiated mutual information I(x;\pi(x)), where x is a random 
    variable instantiation and \pi(x) is a parent set instantiation, i.e. the parent r.v. 
    instantiations that this instantiation of x is conditioned.

    Reference Equation:
        I(x;\pi(x)) = p(x, \pi(x)) \times \log_2[ \frac{ p(x, \pi(x)) }{ p(x) p(\pi(x)) } ]

    
    Args:
        :param p_x: The joint probability value p(X=x).
        :type p_x: float
        :param p_xp: The joint probability value p(X=x, \pi(X)=\pi(x)).
        :type p_xp: float
    
    Kwargs:
        :param p_p: The joint probability value p(\pi(X) = \pi(x)).
        :type p_p: float
    """
    if p_xp == 0:
        return 0
    try:
        mi = p_xp * np.log2(p_xp / (p_x * p_p))
    except ZeroDivisionError:
        raise InvalidProbabilityError(p_xp, p_x=p_x, p_p=p_p)
    return mi

def mutual_info_from_probs(joint_prob_matrix:np.array):
    """ Function that takes a joint probability matrix (X, \pi(X)) ~ p and calculates the
    r.v. conditional entropy. 
    
    Reference Equation:
        H(X|\pi(X)) = - \sum_{(x, \pi(x) \in X \times \pi(X)} p(x, \pa(x)) \times \log_2[ p(x | \pi(x)) ]
    
    Args
        :param joint_prob_matrix: The joint distribution matrix where the cell value for any i-row
            j-column represents the probability of \pa(x)_i, x_j, i.e., p(\pa(x)_i, x_j).
        :type joint_prob_matrix: np.array

    """
    # Get Marginals
    P_X_marginal = np.sum(joint_prob_matrix, axis=0)
    P_P_marginal = np.sum(joint_prob_matrix, axis=1)
    # Calculate
    mi = 0
    for i, p_xp_row in enumerate(joint_prob_matrix):
        for j, p_xp in enumerate(p_xp_row):
            p_p = P_P_marginal[i]
            p_x = P_X_marginal[j]
            mi += instantiated_mutual_info_from_probs(p_x, p_xp, p_p)
    return mi

def instantiated_mutual_info(
        data:np.array,
        x_state_idx:int,
        parent_state_indices:list,
        store=None,
        data_eff:list=None
        ):
    """ Function to calculate instantiated mutual information from data given feature state
    indices.

    Args:
        :param data: The standard one-hot encoded BKB data.
        :type data: np.array
        :param x_state_idx: The r.v. instantiation X=x index as governed by the data sets feature states list and  
            which is conditioned upon the parent set instantiations.
        :type x_state_idx: int
        :param parent_state_indices: The r.v. instantiations in the set of parent set, \pi(x), and are also indices
            according to the data set feature states list. 
        :type parent_state_indices: list

    Kwargs:
        :param store: The probability store which can and should be updated to save time.
        :type store: dict
        :param data_eff: The bkb data formatted for efficiency as given by
            pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
        :type data_eff: list
    """
    if data_eff is not None:
        p_xp, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx, parent_state_indices, store)
        p_x, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx=x_state_idx, parent_state_indices=None, store=store)
        p_p, store = joint_prob_eff(data_eff, data.shape[0], x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    else:
        p_xp, store = joint_prob(data, x_state_idx, parent_state_indices, store)
        p_x, store = joint_prob(data, x_state_idx=x_state_idx, parent_state_indices=None, store=store)
        p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    if p_xp == 0:
        return 0, store
    return instantiated_mutual_info_from_probs(p_x, p_xp, p_p), store

def mutual_info(
        data:np.array,
        feature_states:list,
        X,
        pa_set,
        store:dict=None,
        data_eff:list=None,
        feature_states_map=None,
        ):
    """ Function to calculate mutual information from bkb data given a feature X and a set of parent
    features pa_X = \pi(X).

    Args:
        :param data: The standard one-hot encoded BKB data.
        :type data: np.array
        :param feature_states: The standard feature_states list for the encoded BKB data.
        :type feature_states: list
        :param X: The random variable that will be continued on, must match a feature in the feature_states list.
        :type X: str,int
        :param pa_set: A set of parent r.v.'s, all parent r.v.'s must match a feature in the feature_states list.
        :type pa_set: list

    Kwargs:
        :param store: The probability store which can and should be updated to save time.
        :type store: dict
        :param data_eff: The bkb data formatted for efficiency as given by
            pybkb.learn.backends.bkb.gobnilp.BKBGobnilpBackend.format_data_for_probs function.
        :type data_eff: list
        :param feature_states_map: A dictionary mapping features (r.v's) to all their respective instantiation 
            indices as given by the function pybkb.utils.probability.build_feature_state_map().  Will be built on demand if not passed.
        :type feature_states_map: dict
    """
    if not feature_states_map:
        feature_states_map = build_feature_state_map(feature_states)
    # Construct joint probability matrix
    joint_prob_matrix = extract_joint_prob_matrix(
            data,
            feature_states_map[X],
            [feature_states_map[pa] for pa in pa_set],
            store=store,
            data_eff=data_eff,
            )
    # Calculate mutual information 
    return mutual_info_from_probs(joint_prob_matrix), feature_states_map

def build_feature_state_map(feature_states, no_state_names=True):
    feature_states_map = defaultdict(list)
    for idx, (feature, state) in enumerate(feature_states):
        if no_state_names:
            feature_states_map[feature].append(idx)
        else:
            feature_states_map[feature].append((idx, state))
    return dict(feature_states_map)

"""
def instantiated_ll(data, x_state_idx, parent_state_indices, store=None):
    p_xp, store = joint_prob(data, x_state_idx, parent_state_indices, store)
    p_p, store = joint_prob(data, x_state_idx=None, parent_state_indices=parent_state_indices, store=store)
    if p_xp == 0:
        return 0, store
    try:
        ll = p_xp * np.log2(p_xp / p_p)
    except MathDomainError:
        ll = 0
    return ll, store

def logl(data, x, parents, feature_states, feature_states_map, store=None):
    if store is None:
        store = build_probability_store()
    ill_table = []
    ll = 0
    parent_state_sets = [[s_idx for s_idx, _ in feature_states_map[p]] for p in parents]
    for x_state_idx, _ in feature_states_map[x]:
        for parent_state_indices in itertools.product(*parent_state_sets):
            ill_row = [x_state_idx] + list(parent_state_indices)
            _ll, store = instantiated_ll(data, x_state_idx, list(parent_state_indices), store)
            ll += _ll
            ill_row.append(_ll)
            ill_table.append(ill_row)
    return ll, format_imi_table(ill_table, feature_states), store
"""


"""
def format_imi_table(table, feature_states):
    temp_list = []
    for row in table:
        temp_row = []
        for i in range(len(row) - 1):
            temp_row.append(feature_states[row[i]])
        temp_row.append(row[-1])
        temp_list.append(temp_row)
    temp_dict = defaultdict(list)
    for row in temp_list:
        for i in range(len(row) - 1):
            temp_dict[row[i][0]].append(row[i][1])
        temp_dict["Mutual Info"].append(row[-1])
    return pd.DataFrame.from_dict(temp_dict)


def load_sprinkler():
    import pickle
    with open('sprinkler.dat', 'rb') as f_:
        data, feature_states, srcs = pickle.load(f_)

    feature_states_map = build_feature_state_map(feature_states)
    return data, feature_states, feature_states_map

def sprinkler():
    data, feature_states, feature_states_map = load_sprinkler()
    w, iw, store = mutual_info(data, 'cloudy', ['rain'], feature_states, feature_states_map)
    print(feature_states)
    print(w)
    print(iw)
    print(store)

#sprinkler()
"""
