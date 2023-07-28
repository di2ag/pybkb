import os
import json
import re

path_bkb = '../nips_experiments/scores/bkb'
path_bn = '../nips_experiments/scores/bn'

datasets_bkb = os.listdir(path_bkb)
datasets_bn = os.listdir(path_bn)

dataset_options = {}
for dataset in set.intersection(*[set(datasets_bkb), set(datasets_bn)]):
    bkb_options = os.listdir(os.path.join(path_bkb, dataset))
    bkb_options = [int(re.split('-|\.', op)[1]) for op in bkb_options if 'scores' in op]
    bn_options = os.listdir(os.path.join(path_bn, dataset))
    bn_options = [int(re.split('-|\.', op)[1]) for op in bn_options if 'scores' in op]
    try:
        highest_common_palim = max(set.union(*[set(bkb_options), set(bn_options)]))
        dataset_options[dataset] = highest_common_palim
    except:
        continue

with open('../nips_experiments/scores/palim_common_options.json', 'w') as f_:
    json.dump(dataset_options, f_, indent=2)
