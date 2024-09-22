import os
import argparse
import json
import numpy as np
import data_utils
import pandas as pd
import torch
import pickle as pkl

from  data_utils import align_mol_to_frags, to_graph_mol, split_by_continuity, find_indices
from util.chemutils import *
from util.mol_tree import MolTree
from tqdm import tqdm
from rgcn.data_preprocess import load_graph



def preprocess_data(smiles_list,  task_name='Mutagenicity'):
    processed_data = []
    error = []
    for i, smi_mol in enumerate(smiles_list):

        mol = get_mol(smi_mol)
        nodes_out, edges_out = to_graph_mol(mol, task_name)   # 输入分子


        if len(edges_out) <= 0:
            error.append(i)
            continue


        processed_data.append({
            # 'graph_in': fragment_edges,
            'graph_out': edges_out,
            # 'node_features_in': nodes_out,
            'node_features_out': nodes_out,
            'smiles_out': smi_mol,
            # 'exit_points': exit_points,
            # 'clique':clique
        })

    print('error: ' + str(len(error)))
    return  processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--save_fold', type=str, default='./data_preprocessed/hERG/', help='path to save data')
    parser.add_argument('--name', type=str, default='data', help='name of dataset')

    parser.add_argument('--max_atoms', type=int, default=89, help='maximum atoms of generated mol')
    parser.add_argument('--atom_types', type=int, default=14, help='number of atom types in generated mol')
    parser.add_argument('--edge_types', type=int, default=4, help='number of edge types in generated mol')
    # parser.add_argument('--bin_path', type=str, default='./log/BBBP_logs/rgcn_data/dgl_graph.bin')
    # parser.add_argument('--group_path', type=str, default='./log/BBBP_logs/rgcn_data/BBBP_group.csv')
    # parser.add_argument('--bin_path', type=str, default='./log/HIV_logs/rgcn_data/dgl_graph.bin')
    # parser.add_argument('--group_path', type=str, default='./log/HIV_logs/rgcn_data/HIV_group.csv')
    # parser.add_argument('--bin_path', type=str, default='./log/ESOL_logs/rgcn_data/dgl_graph.bin')
    # parser.add_argument('--group_path', type=str, default='./log/ESOL_logs/rgcn_data/ESOL_group.csv')
    parser.add_argument('--bin_path', type=str, default='./log/hERG_logs/rgcn_data/dgl_graph.bin')
    parser.add_argument('--group_path', type=str, default='./log/hERG_logs/rgcn_data/hERG_group.csv')

    args = parser.parse_args()

    print('start loading data...')
    task_name = 'hERG'

    # 加载数据
    with open('./rgcn/result/hyperparameter_{}.pkl'.format(task_name), 'rb') as f:
        hyperparameter = pkl.load(f)

    rgcn_hidden_feats = hyperparameter['rgcn_hidden_feats']
    ffn_hidden_feats = hyperparameter['ffn_hidden_feats']
    classification = hyperparameter['classification']

    train_set, val_set, test_set, task_number = load_graph(
        bin_path=args.bin_path,
        group_path=args.group_path,
        classification=classification,
        seed=2024,
        random_shuffle=False)

    print('start preprocessing...')
    smiles_lst = []
    sum = 0
    for data_item in tqdm(train_set):
        smiles, g_rgcn, labels, smask, sub_name = data_item

        mol = Chem.MolFromSmiles(smiles)
        atom_num = mol.GetNumAtoms()
        if atom_num <= 89:
            smiles_lst.append(smiles)
    for data_item in tqdm(val_set):
        smiles, g_rgcn, labels, smask, sub_name = data_item
        mol = Chem.MolFromSmiles(smiles)
        atom_num = mol.GetNumAtoms()
        if atom_num <= 89:
            smiles_lst.append(smiles)
    # sum = 0
    for data_item in tqdm(test_set):
        smiles, g_rgcn, labels, smask, sub_name = data_item
        mol = Chem.MolFromSmiles(smiles)
        atom_num = mol.GetNumAtoms()
        if '.' in smiles:
            sum += 1
        # if atom_num <= 89:
        #     # print(labels)
        #     sum +=1
    #         # smiles_lst.append(smiles)

    print(sum)


    # 处理数据
    processed_data = preprocess_data(smiles_lst, task_name=task_name)

    if not os.path.exists(args.save_fold):
        os.makedirs(args.save_fold)
    with open(args.save_fold + args.name + '.json', 'w') as f:
        json.dump(processed_data, f)

    # convert the processed data to Adjacency Matrix
    full_nodes = []
    full_edges = []
    # frag_nodes = []
    # frag_edges = []
    # gen_len = []
    # exit_point = []
    full_smi = []
    # frag = []
    # clique = []

    for line in tqdm(processed_data):

        full_node = torch.zeros([args.max_atoms, args.atom_types]) # (89, 14)
        for i in range(len(line['node_features_out'])):
            for j in range(len(line['node_features_out'][0])):
                full_node[i][j] = line['node_features_out'][i][j]
        full_nodes.append(full_node)
        # print(full_node.shape)

        full_edge = torch.zeros([args.edge_types, args.max_atoms, args.max_atoms]) # (4, 89, 89)
        for i in (line['graph_out']):
            start=i[0]
            end=i[2]
            edge=i[1]
            full_edge[edge,start,end]=1.0
            full_edge[edge,end,start]=1.0
        full_edges.append(full_edge)
    #
    #
    #     # # input fragments
    #     # frag_node = torch.zeros([args.max_atoms, args.atom_types])  # (89, 14)
    #     # for i in range(len(line['node_features_in'])):
    #     #     for j in range(len(line['node_features_in'][0])):
    #     #         frag_node[i][j] = line['node_features_in'][i][j]
    #     #
    #     # frag_nodes.append(frag_node)
    #     # frag_edge = torch.zeros([args.edge_types, args.max_atoms, args.max_atoms])  # (4, 89, 89)
    #     # for i in (line['graph_in']):
    #     #     start = i[0]
    #     #     end = i[2]
    #     #     edge = i[1]
    #     #     frag_edge[edge, start, end] = 1.0
    #     #     frag_edge[edge, end, start] = 1.0
    #     # frag_edges.append(frag_edge)
    #
    #     # gen_len.append(len(line['clique']))
    #     # if len(line['exit_points']) != 1:
    #     #     print(line['smiles_out'])
    #
    #     # exit_point.append(line['exit_points'])
        full_smi.append(line['smiles_out'])
    #     # clique.append(line['clique'])

    full_nodes = torch.tensor([item.detach().numpy() for item in full_nodes])
    full_edges = torch.tensor([item.detach().numpy() for item in full_edges])
    # # frag_nodes = torch.tensor([item.detach().numpy() for item in frag_nodes])
    # # frag_edges = torch.tensor([item.detach().numpy() for item in frag_edges])
    #
    np.save(args.save_fold + 'full_nodes', full_nodes)
    np.save(args.save_fold + 'full_edges', full_edges)
    # np.save(args.save_fold + 'frag_nodes', frag_nodes)
    # np.save(args.save_fold + 'frag_edges', frag_edges)
    # np.save(args.save_fold + 'gen_len', gen_len)
    # np.save(args.save_fold + 'exit_point', exit_point)
    np.save(args.save_fold + 'full_smi', full_smi)
    # clique = np.array(clique, dtype=object)
    # np.save(args.save_fold + 'clique', clique)

    fp = open(args.save_fold + 'config.txt', 'w')
    config = dict()
    config['atom_list'] = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'}
    config['node_dim'] = args.atom_types
    config['max_size'] = args.max_atoms
    config['bond_dim'] = args.edge_types
    fp.write(str(config))
    fp.close()

    print('done!')




