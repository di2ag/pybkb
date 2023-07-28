import os
import csv
import numpy as np
import compress_pickle
import itertools
import tqdm
from collections import defaultdict
from operator import itemgetter


DATASETS = [
        ('TCGA-BRCA', '/home/public/data/ncats/TCGA-BRCA/mutations.csv'),
        #('TCGA-GBM', '/home/public/data/ncats/TCGA-GBM/mutations.csv'),
        #('TCGA-OV', '/home/public/data/ncats/TCGA-OV/mutations.csv'),
        #('TCGA-LUAD', '/home/public/data/ncats/TCGA-LUAD/mutations.csv'),
        #('TCGA-UCEC', '/home/public/data/ncats/TCGA-UCEC/mutations.csv'),
        ]

SAVE_PATH = '../data/tcga'

for name, dataset in DATASETS:
    with open(dataset, 'r') as f_:
        reader = csv.reader(f_, delimiter=',')
        rows = []
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            info = {feature: state for feature, state in zip(header, row)}
            rows.append(info)
    # Filter on genes and chromesome-start positions
    unique_start_pos = defaultdict(int)
    unique_genes = defaultdict(int)
    for row in rows:
        unique_start_pos[f'{row["Hugo_Symbol"]}-{row["Chromosome"]}:{row["Start_Position"]}'] += 1
        unique_genes[row["Hugo_Symbol"]] += 1
    # Filter based on frequency of appreance
    filter_starts = []
    filter_genes = []
    for pos, count in unique_start_pos.items():
        if count > 2:
            filter_starts.append(pos)
    for gene, count in unique_genes.items():
        if count > 10:
            filter_genes.append(gene)
    print('unique genes', len(filter_genes))
    print('unique pos', len(filter_starts))
    # Construct database
    data_genes = defaultdict(list)
    data_pos = defaultdict(list)
    data_both = defaultdict(list)
    for row in rows:
        try:
            pat_id = row.get("Patient.ID", None)
            if pat_id is None:
                pat_id = row.get("Patient ID", None)
            if pat_id is None:
                raise KeyError
            gene = row["Hugo_Symbol"]
            pos = f'{row["Hugo_Symbol"]}-{row["Chromosome"]}:{row["Start_Position"]}'
            # Get mutational state (Reference Allele, Tumor_Seq_Allele1, Tumor_Seq_Allele2)
            pos_state = (
                    row["Reference_Allele"],
                    row["Tumor_Seq_Allele1"],
                    row["Tumor_Seq_Allele2"],
                    )
            gene_state = row["Variant_Classification"]
            if gene in filter_genes and "Silent" not in gene_state:
                data_genes[pat_id].append((gene, gene_state))
                if pos not in filter_starts:
                    data_both[pat_id].append((gene, gene_state))
            if pos in filter_starts:
                data_pos[pat_id].append((pos, pos_state))
                data_both[pat_id].append((pos, pos_state))
        except Exception as ex:
            print(row)
            raise ex
    data_genes_new = {}
    # Go through and form multiple gene mutations into frozensets
    for pat_id, feature_states in data_genes.items():
        fs_dict = defaultdict(list)
        for f, s in feature_states:
            fs_dict[f].append(s)
        new_feature_states = []
        for f, ss in fs_dict.items():
            if len(set(ss)) == 1:
                new_feature_states.append((f, ss[0]))
            else:
                new_feature_states.append((f, frozenset(ss)))
        data_genes_new[pat_id] = new_feature_states
    data_both_new = {}
    for pat_id, feature_states in data_both.items():
        fs_dict = defaultdict(list)
        for f, s in feature_states:
            fs_dict[f].append(s)
        new_feature_states = []
        for f, ss in fs_dict.items():
            if len(set(ss)) == 1:
                new_feature_states.append((f, ss[0]))
            else:
                new_feature_states.append((f, frozenset(ss)))
        data_both_new[pat_id] = new_feature_states
    # Filter on gene states
    unique_gene_states = defaultdict(int)
    for pat_id, feature_states in data_genes_new.items():
        for fs in feature_states:
            unique_gene_states[fs] += 1
    data_genes = {}
    for pat_id, feature_states in data_genes_new.items():
        new_feature_states = []
        for fs in feature_states:
            if unique_gene_states[fs] > 10:
                new_feature_states.append(fs)
        data_genes[pat_id] = new_feature_states
    data_both = {}
    for pat_id, feature_states in data_both_new.items():
        new_feature_states = []
        for fs in feature_states:
            if fs not in unique_gene_states:
                new_feature_states.append(fs)
            elif unique_gene_states[fs] > 10:
                new_feature_states.append(fs)
        data_both[pat_id] = new_feature_states
    # Construct bkb data
    feature_states_genes = list(set.union(*[set(genes) for _, genes in data_genes.items()]))
    feature_states_pos = list(set.union(*[set(poss) for _, poss in data_pos.items()]))
    feature_states_both = list(set.union(*[set(both) for _, both in data_both.items()]))
    # Filter out features that only have one state.
    fs_dict_genes = defaultdict(list)
    for f,s in feature_states_genes:
        fs_dict_genes[f].append(s)
    fs_dict_pos = defaultdict(list)
    for f,s in feature_states_pos:
        fs_dict_pos[f].append(s)
    fs_dict_both = defaultdict(list)
    for f,s in feature_states_both:
        fs_dict_both[f].append(s)
    '''
    feature_states_genes = []
    for f, ss in fs_dict_genes.items():
        if len(ss) > 1:
            feature_states_genes.extend([(f,s) for s in ss])
    feature_states_pos = []
    for f, ss in fs_dict_pos.items():
        if len(ss) > 1:
            feature_states_pos.extend([(f,s) for s in ss])
    feature_states_both = []
    for f, ss in fs_dict_both.items():
        if len(ss) > 1:
            feature_states_both.extend([(f,s) for s in ss])
    '''
    # Go through and add no mutation if case did not have a mutation
    feature_genes = set([f for f, _ in feature_states_genes])
    _data_genes = {}
    for pat_id, feature_states in data_genes.items():
        fs_dict = {f: s for f, s in feature_states}
        new_feature_states = []
        for f in feature_genes:
            if f not in fs_dict:
                new_feature_states.append((f, 'Not_Mutated'))
                continue
            new_feature_states.append(fs_dict[f])
        _data_genes[pat_id] = new_feature_states
    feature_pos = set([f for f, _ in feature_states_pos])
    _data_pos = {}
    for pat_id, feature_states in data_pos.items():
        fs_dict = {f: s for f, s in feature_states}
        new_feature_states = []
        for f in feature_pos:
            if f not in fs_dict:
                new_feature_states.append((f, 'Not_Mutated'))
                continue
            new_feature_states.append(fs_dict[f])
        _data_pos[pat_id] = new_feature_states
    feature_both = set([f for f, _ in feature_states_both])
    _data_both = {}
    for pat_id, feature_states in data_both.items():
        fs_dict = {f: s for f, s in feature_states}
        new_feature_states = []
        for f in feature_both:
            if f not in fs_dict:
                new_feature_states.append((f, 'Not_Mutated'))
                continue
            new_feature_states.append((f, fs_dict[f]))
        _data_both[pat_id] = new_feature_states
    data_genes = _data_genes
    data_pos = _data_pos
    data_both = _data_both
    feature_states_genes = list(set.union(*[set(genes) for _, genes in data_genes.items()]))
    feature_states_pos = list(set.union(*[set(poss) for _, poss in data_pos.items()]))
    feature_states_both = list(set.union(*[set(both) for _, both in data_both.items()]))
    fs_map_genes = {fs: idx for idx, fs in enumerate(feature_states_genes)}
    fs_map_pos = {fs: idx for idx, fs in enumerate(feature_states_pos)}
    fs_map_both = {fs: idx for idx, fs in enumerate(feature_states_both)}
    print('gene feature_states', len(feature_states_genes))
    print('pos feature states', len(feature_states_pos))
    print('both feature states', len(feature_states_both))
    fs_dict_both = defaultdict(list)
    for f,s in feature_states_both:
        fs_dict_both[f].append(s)
    print(fs_dict_both)
    print(np.prod([len(ss) for f, ss in fs_dict.items()]))
    # Build for genes
    srcs_genes = []
    data_np_genes = []
    for pat_id, feature_states in data_genes.items():
        srcs_genes.append(pat_id)
        row = np.zeros(len(feature_states_genes))
        for fs in feature_states:
            row[fs_map_genes[fs]] = 1
        data_np_genes.append(row)
    data_np_genes = np.array(data_np_genes)
    # Build for pos
    srcs_pos = []
    data_np_pos = []
    for pat_id, feature_states in data_pos.items():
        if len(feature_states) == 0:
            continue
        srcs_pos.append(pat_id)
        row = np.zeros(len(feature_states_pos))
        for fs in feature_states:
            row[fs_map_pos[fs]] = 1
        data_np_pos.append(row)
    data_np_pos = np.array(data_np_pos)
    print('pos data dim', data_np_pos.shape)
    # Build for both
    srcs_both = []
    data_np_both = []
    for pat_id, feature_states in tqdm.tqdm(data_both.items()):
        if len(feature_states) == 0:
            continue
        # Need to potentially expand if genes had more than one mutation
        fs_dict = defaultdict(list)
        for f, s in feature_states:
            fs_dict[f].append(s)
        features = list(fs_dict.keys())
        state_combos = [fs_dict[f] for f in features]
        total = np.prod([len(ss) for ss in state_combos])
        for idx, state_config in tqdm.tqdm(enumerate(itertools.product(*state_combos)), total=total, leave=False):
            srcs_both.append(f'{pat_id}-{idx}')
            row = np.zeros(len(feature_states_both))
            for f, s in zip(features, state_config):
                row[fs_map_both[(f,s)]] = 1
            data_np_both.append(row)
    data_np_both = np.array(data_np_both)
    print(data_np_both.shape)
    print(feature_states_both)
    # Save off
    save_path = os.path.join(SAVE_PATH, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, 'gene_mutational.dat'), 'wb') as f_:
        compress_pickle.dump((data_np_genes, feature_states_genes, srcs_genes), f_, compression='lz4')
    with open(os.path.join(save_path, 'position_mutational.dat'), 'wb') as f_:
        compress_pickle.dump((data_np_pos, feature_states_pos, srcs_pos), f_, compression='lz4')
    with open(os.path.join(save_path, 'gene_position_mutational.dat'), 'wb') as f_:
        compress_pickle.dump((data_np_both, feature_states_both, srcs_both), f_, compression='lz4')
