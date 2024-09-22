import os
import argparse
import json
import numpy as np
import data_utils
import pandas as pd
import torch

from  data_utils import align_mol_to_frags, to_graph_mol, split_by_continuity, find_indices
from util.chemutils import *
from util.mol_tree import MolTree


def get_motif(smi):
    mol = get_mol(smi)

    tree = MolTree(smi)
    leaf_cliques = []
    ring_cliques = []
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
    # print(ssr)
    for node in tree.nodes:
        # print(node.clique)
        if node.is_leaf :
            if node.clique in ssr:
                ring_cliques.append(node.clique)
            else:
                leaf_cliques.append(node.clique)

        else:
            if node.clique in ssr:
                ring_cliques.append(node.clique)
    # print(leaf_cliques)

    # 得到和叶子节点相连的exit_vector  exit_vector:leaf
    mutate = []
    for c in leaf_cliques:
        exit_vector = []
        for atom_idx in c:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            for nei in neighbors:
                neighbor_idx = nei.GetIdx()
                if neighbor_idx not in c:
                    exit_vector.append(neighbor_idx)

        cliques = split_by_continuity(c)

        if len(cliques) == 1:
            for i in range(len(exit_vector)):
                if exit_vector[i] > max(c):
                    exit_vector[i] = exit_vector[i] - len(c)
        else:
            for i in range(len(exit_vector)):
                if exit_vector[i] > min(c):
                    exit_ind = find_indices(cliques, exit_vector[i])
                    if exit_ind == 0:
                        exit_vector[i] = exit_vector[i] - len(cliques[exit_ind])
                    else:
                        len2 = 0
                        for j in range(exit_ind + 1):
                            len2 += len(cliques[j])
                        exit_vector[i] = exit_vector[i] - len2

        mutate.append(['leaf', c, exit_vector])

    for ring in ring_cliques:
        mutate.append(['ring', ring])

    return mutate


def data_wash(data_path):
    data = pd.read_csv(data_path)
    smiles_list = data['smiles'].tolist()
    smiles_list = list(set(smiles_list))

    error = []
    for s in smiles_list:

        try:
            mutate = get_motif(s)

            for mute in mutate:
                # print(mute)
                if mute[0] == 'leaf':
                    if len(mute[2]) > 1:
                        error.append(s)
                        break
        except:
            print('could not tree')
            error.append(s)


    cant_to_adj = []
    for s in smiles_list:
        mol = get_mol(s)

        nodes_out, edges_out = to_graph_mol(mol, 'Mutagenicity')
        if len(edges_out) <= 0:
            cant_to_adj.append(s)
    filter_data =list(set(error).union(set(cant_to_adj)))
    print(len(error))
    print(len(cant_to_adj))
    print(len(filter_data))
    df_filtered = data[~data['smiles'].isin(filter_data)]
    # df_filtered.to_csv('./Mutagenicity/Mutagenicity.csv', index=False)
    df_filtered.to_csv(data_path, index=False)


def dataset_wash(data_path, task_name):
    # 清除无机分子， 混合物， 重复的分子
    # parameter
    # load data set
    # data = pd.read_csv('{}.csv'.format(task_name))
    data = pd.read_csv(data_path)
    origin_data_num = len(data)

    # remove molecule can't processed by rdkit_des
    print('********dealing with compounds with rdkit_des*******')
    smiles_list = data['smiles'].values.tolist()
    cant_processed_smiles_list = []
    for index, smiles in enumerate(smiles_list):
        if index % 10000 == 0:
            print(index)
        try:
            molecule = Chem.MolFromSmiles(smiles)
            smiles_standard = Chem.MolToSmiles(molecule)
            data['smiles'][index] = smiles_standard
        except:
            cant_processed_smiles_list.append(smiles)
            data.drop(index=index, inplace=True)
    print("compounds can't be processed by rdkit_des: {} molecules, {}\n".format(len(cant_processed_smiles_list),
                                                                      cant_processed_smiles_list))

    # remove mixture and salt
    print('********dealing with inorganic compounds*******')
    data = data.reset_index(drop=True)
    smiles_list = data['smiles'].values.tolist()
    mixture_salt_list = []
    for index, smiles in enumerate(smiles_list):
        if index % 10000==0:
            print(index)
        symbol_list = list(smiles)
        if '.' in symbol_list:
            mixture_salt_list.append(smiles)
            data.drop(index=index, inplace=True)
    print('inorganic compounds: {} molecules, {}\n'.format(len(mixture_salt_list), mixture_salt_list))


    # remove inorganic compounds
    print('********dealing with inorganic compounds*******')
    data = data.reset_index(drop=True)
    smiles_list = data['smiles'].values.tolist()
    inorganics = []
    atom_list = []
    for index, smiles in enumerate(smiles_list):
        if index % 10000==0:
            print(index)
        mol = Chem.MolFromSmiles(smiles)
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                break
            else:
                count += 1
        if count == mol.GetNumAtoms():
            inorganics.append(smiles)
            data.drop(index=index, inplace=True)
    print('inorganic compounds: {} molecules, {}\n'.format(len(inorganics), inorganics))

    print('********dealing with duplicates*******')
    consistent_mol = ['smiles', task_name]
    print('duplicates:{} molecules {}\n'.format(len(data[data.duplicated(consistent_mol)]['smiles'].values),
                                                 data[data.duplicated(consistent_mol)]['smiles'].values))
    data.drop_duplicates(consistent_mol, keep='first', inplace=True)
    consistent_mol_2 = ['smiles']
    print('duplicates and conflict:{} molecules {}\n'.format(len(data[data.duplicated(consistent_mol_2)]['smiles'].values),
                                                                 data[data.duplicated(consistent_mol_2)]['smiles'].values))
    data.drop_duplicates(consistent_mol_2, keep=False, inplace=True)
    print('Data washing is over!')

    import random
    data = data[['smiles', task_name]]
    len_data = len(data)
    group_list = ['training' for x in range(int(len_data*0.8))] + ['valid' for j in range(int(len_data*0.1))] + ['test' for i in range(len_data - int(len_data*0.8)-int(len_data*0.1))]
    random.shuffle(group_list)
    data['group'] = group_list
    print("{} to {} after datawash.".format(origin_data_num, len_data))
    data.to_csv('{}.csv'.format(task_name), index=False)


if __name__ == '__main__':
    print('start loading data...')
    # task_name = 'Mutagenicity'
    # task_list = ['ESOL','BBBP', 'hERG', 'Mutagenicity']
    task_list = ['HIV']

    for task in task_list:
        data_path = './' +task + '/' +task+ '.csv'
        # print(data_path)
        print('start preprocessing...')
        wash_data = data_wash(data_path)

        # data = dataset_wash(data_path,task)


    # print('start preprocessing...')
    # # 清洗数据 不能转换为图数据的 没有叶节点的 不能构建分子树的
    # # wash_data = data_wash(data_path)
    #
    # df = pd.read_csv('./Mutagenicity/Mutagenicity.csv')
    # test_data = df[df['group'] == 'test'][:200]
    # test_data.to_csv('./Mutagenicity/Mutagenicity_test.csv', index=False)
    # smiles = 'c1ccc2c(OCC3CO3)nsc2c1'
    # mol = get_mol(smiles)
    # nodes_out, edges_out = to_graph_mol(mol, 'Mutagenicity')
    # print(edges_out)
    # data = pd.read_csv(data_path)
    # smiles_list = data['smiles'].tolist()
    # smiles = list(set(smiles_list))
    # print(len(smiles))

