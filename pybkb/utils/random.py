import random
import networkx as nx
from networkx.exception import NetworkXNoCycle

from pybkb.bkb import BKB
from pybkb.exceptions import BKBNotMutexError


def generate_random_bkb_inference(
        features:dict,
        pa_lim:int=2,
        random_seed=None,
        ):
    """ A generator function that will generate a random bkb inference, meaning a BKB that 
    has at most one RV instantiation and is acyclic. 

    Args:
    :param features: A dictionary with RV names as keys, and list values delineated all 
        possible RV states (instantiations).
    :type features: dict
    
    Kwargs:
    :param pa_lim: Maximum number of I-nodes in any S-node tail.
    :type pa_lim: int
    :param random_seed: A random seed used in initialize the random generator.
    :type int:
    """
    # Initialize
    if random_seed:
        random.seed(random_seed)
    # Collect random instantiations
    world = set()
    for feature, states in features.items():
        rand_state = random.choice(states)
        world.add((feature, rand_state))
    # Iterate until we find an acyclic BKB with all instantiations
    while True:
        bki = BKB()
        # Add I-nodes
        for feature, state in world:
            bki.add_inode(feature, state)
        # Generate some random S-nodes
        processed_heads = set()
        while len(processed_heads) < len(world):
            rand_snode_head = random.choice(list(world - processed_heads))
            pa_num = random.randrange(0, pa_lim + 1)
            if pa_num == 0:
                bki.add_snode(
                        rand_snode_head[0],
                        rand_snode_head[1],
                        prob=random.random(),
                        )
            else:
                rand_snode_tail = list(random.choices(list(world), k=pa_num))
                bki.add_snode(
                        rand_snode_head[0],
                        rand_snode_head[1],
                        prob=random.random(),
                        tail=rand_snode_tail,
                        )
            processed_heads.add(rand_snode_head)
        # Check if BKB inference is valid, i.e. no cycles.
        try:
            _ = nx.find_cycle(bki.construct_nx_graph())
        except NetworkXNoCycle:
            break
    return bki


def generate_random_bkb_inferences(
        features:dict,
        k:int,
        pa_lim:int=2,
        random_seed:int=None,
        ):
    """ A generator function that will generate k random bkb inferences, meaning a
    set of BKBs that has at most one RV instantiation and is acyclic. 

    Args:
    :param features: A dictionary with RV names as keys, and list values delineated all 
        possible RV states (instantiations).
    :type features: dict
    :param k: Number of BKB inferences to generate.
    :type k: int
    
    Kwargs:
    :param pa_lim: Maximum number of I-nodes in any S-node tail.
    :type pa_lim: int
    :param random_seed: A random seed used in initialize the random generator.
    :type int:
    """
    # Initialize
    if random_seed:
        random.seed(random_seed)
    bkis = []
    for _ in range(k):
        bkis.append(generate_random_bkb_inference(features, pa_lim))
    return bkis


def generate_random_bkb(
        features:dict,
        k:int,
        pa_lim:int=2,
        random_seed:int=None,
        ):
    """ A generator function that will generate a random bkb with k number of inferences.

    Args:
    :param features: A dictionary with RV names as keys, and list values delineated all 
        possible RV states (instantiations).
    :type features: dict
    :param k: Number of BKB inferences to generate.
    :type k: int
    
    Kwargs:
    :param pa_lim: Maximum number of I-nodes in any S-node tail.
    :type pa_lim: int
    :param random_seed: A random seed used in initialize the random generator.
    :type int:
    """
    # Initialize
    if random_seed:
        random.seed(random_seed)
    bkis = []
    for _ in range(k):
        if len(bkis) == 0:
            bkis.append(generate_random_bkb_inference(features, pa_lim))
            continue
        while True:
            found = [False for _ in range(len(bkis))]
            rand_bki = generate_random_bkb_inference(features, pa_lim) 
            for i, bki in enumerate(bkis):
                try:
                    _ = BKB.union(*[bki, rand_bki]).is_mutex()
                    found[i] = True
                except BKBNotMutexError:
                    break
            if all(found):
                break
        bkis.append(rand_bki)
    return BKB.union(*bkis), bkis
