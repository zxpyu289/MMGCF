import sys
import os
import csv
import numpy as np
import networkx as nx
import random
import pandas as pd
sys.path.insert(0,'..')
import torch
import torch.nn.functional as F
import argparse

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMMPA
from rdkit.Chem import rdMolAlign
from util.mol_tree import MolTree
from itertools import product
from util.chemutils import *

bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
atom_types = {0:'Br1(0)', 1:'C4(0)', 2:'Cl1(0)', 3:'F1(0)', 4:'H1(0)', 5:'I1(0)',6:'N2(-1)', 7:'N3(0)', 8:'N4(1)', 9:'O1(-1)', 10:'O2(0)', 11:'S2(0)', 12:'S4(0)', 13:'S6(0)'}
num2symbol = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'}


def get_mask_index(smask_index_list, smi_index):
    smask_index_i = [smask_index_list[i] for i in smi_index]
    return smask_index_i

def dataset_info(dataset):
    if dataset == 'qm9':
        return {'atom_types': ["H", "C", "N", "O", "F"],
                'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
                }
    else:
        return {'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                               'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'],
                'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
                                    13: 6, 14: 3},
                'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                                   10: 'O', 11: 'S', 12: 'S', 13: 'S'},
                'bucket_sizes': np.array(
                    [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58, 84])
                }


def add_atoms(mol, node_list):
    for number in node_list:
        new_atom = Chem.Atom(num2symbol[number])
        charge_num=int(atom_types[number].split('(')[1].strip(')'))
        new_atom.SetFormalCharge(charge_num)
        mol.AddAtom(new_atom)


def decode(tree):
    tree.recover()

    cur_mol = copy_edit_mol(tree.nodes[0].mol)
    global_amap = [{}] + [{} for node in tree.nodes]
    global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

    dfs_assemble(cur_mol, global_amap, [], tree.nodes[0], None)

    cur_mol = cur_mol.GetMol()
    cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
    set_atommap(cur_mol)
    dec_smiles = Chem.MolToSmiles(cur_mol)

    return dec_smiles


def split_by_continuity(numbers):
    if not numbers:
        return []

    numbers.sort()  # 确保数字是按顺序排列的
    result = []
    current_group = [numbers[0]]

    for num in numbers[1:]:
        if num == current_group[-1] + 1:
            current_group.append(num)
        else:
            result.append(current_group)
            current_group = [num]

    result.append(current_group)
    return result


def find_indices(cliques, start):
    ind = None
    for i, c in enumerate(cliques):  # 遍历每个clique及其索引
         if all(element < start for element in c):   # 遍历clique中的元素及其索引
             ind = i

    return ind


def recon_frag_mol(smile, clique):
    # print(clique)

    # cliques = split_by_continuity(clique)
    # print(len(cliques))
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_atoms', type=int, default=89, help='maximum atoms of generated mol')
    parser.add_argument('--atom_types', type=int, default=14, help='number of atom types in generated mol')
    parser.add_argument('--edge_types', type=int, default=4, help='number of edge types in generated mol')

    args = parser.parse_args()

    num2bond =  {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
    num2symbol = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S', 12: 'S', 13: 'S'}


    mol = Chem.MolFromSmiles(smile)
    nodes, edges = to_graph_mol(mol, 'Mutagenicity')

    # print(type(nodes))

    fragment_edges = []
    for i in (edges):
        start = i[0]
        end = i[2]
        edge = i[1]
        if start in clique or end in clique:
            continue

        if start > max(clique):
            start = start - len(clique)
        if end > max(clique):
            end = end - len(clique)
            fragment_edges.append((start, edge, end))

        # if len(cliques) == 1:
        #     if start > max(clique):
        #         start = start - len(clique)
        #     if end > max(clique):
        #         end = end - len(clique)
        # else:
        #     part_clique = []
        #
        #     for i, c in enumerate(cliques):
        #         # print(i)
        #         part_clique.extend(c)
        #
        #         if start > max(part_clique):
        #             start = start - len(part_clique)
        #         if end > max(c):
        #             end = end - len(part_clique)

    # print(frag_edges)


    full_nodes = []
    frag_edges = []
    full_node = torch.zeros([args.max_atoms, args.atom_types])  # (89, 14)
    for i in range(len(nodes)):
        for j in range(len(nodes[0])):
            full_node[i][j] = nodes[i][j]
    full_nodes.append(full_node)

    frag_edge = torch.zeros([args.edge_types, args.max_atoms, args.max_atoms])  # (4, 89, 89)
    for i in (fragment_edges):
        start = i[0]
        end = i[2]
        edge = i[1]
        frag_edge[edge, start, end] = 1.0
        frag_edge[edge, end, start] = 1.0
    frag_edges .append(frag_edge)

    full_nodes = [item.detach().numpy() for item in full_nodes]
    frag_edges =[item.detach().numpy() for item in frag_edges]

    for idx in range(len(full_nodes)):
        node_symbol = []
        # print(frag_nodes)
        for i, atom in enumerate(full_nodes[idx]):
            # print(i)
            if i in clique: continue
            if max(atom) > 0:
                # print(atom)
                node_symbol.append(np.argmax(atom))
        # print(node_symbol)
        rw_mol = Chem.RWMol()

        for i_c, number in enumerate(node_symbol):
            new_atom = Chem.Atom(num2symbol[number])
            charge_num = int(atom_types[number].split('(')[1].strip(')'))
            new_atom.SetFormalCharge(charge_num)
            rw_mol.AddAtom(new_atom)

        for bond in range(3):  # (4, 89, 89)
            for start in range(89):
                for end in range(start + 1, 89):
                    if frag_edges[idx][bond][start][end] == 1:
                        rw_mol.AddBond(start, end, num2bond[bond])


def align_mol_to_frags(smi_molecule, task_name='Mutagenicity'):
    try:
        mol = Chem.MolFromSmiles(smi_molecule)

        motif_mask_index = [x for x in np.load('./rgcn/prediction/motif/{}_motif_1_train_mask_index.npy'.format(task_name), allow_pickle=True)] + \
                           [x for x in np.load('./rgcn/prediction/motif/{}_motif_1_val_mask_index.npy'.format(task_name), allow_pickle=True)] + \
                           [x for x in np.load('./rgcn/prediction/motif/{}_motif_1_test_mask_index.npy'.format(task_name), allow_pickle=True)]
        # 加载attribution数据
        # attribution_motif = pd.read_csv('./rgcn/prediction/attribution/{}_{}_attribution_summary.csv'.format(task_name, 'motif'))
        attribution_motif = pd.read_csv('./rgcn/prediction/attribution/{}_{}_attribution_summary.csv'.format(task_name, 'motif'))
        motif_index_i = attribution_motif[attribution_motif['smiles'] == smi_molecule].index.tolist()

        # 获取attribution中smiles_i的motif_index
        motif_mask_index_i = get_mask_index(motif_mask_index, motif_index_i)
        # print(motif_mask_index_i)
        motif_attribution = attribution_motif[attribution_motif['smiles'] == smi_molecule].attribution_normalized.tolist()

        # 构建motif-attribution 一一对应的字典

        motif_attribution_dict = {}
        for idx, mask in enumerate(motif_mask_index_i):
            mask = tuple(mask) if isinstance(mask, list) else mask
            attribution = motif_attribution[idx]
            motif_attribution_dict[mask] = attribution
            # print(motif_attribution_dict)
        # print(motif_attribution_dict)

        mol_tree = MolTree(smi_molecule)
        leaf_cliques = []
        for node in mol_tree.nodes:
            # print(node.smiles)
            if node.is_leaf:
                leaf_cliques.append(node.clique)
        # print(leaf_cliques)

        leaf_attribution = []
        for c in leaf_cliques:
            c = tuple(c) if isinstance(c, list) else c
            # 比较c和motif_attribution的键 得到c的attibution
            if c in motif_attribution_dict:
                # print(motif_attribution_dict[c])
                leaf_attribution.append(motif_attribution_dict[c])
            else:
                leaf_attribution.append([0.0])

        # print(leaf_attribution)
        max_index = leaf_attribution.index(max(leaf_attribution))
        cut_clique = leaf_cliques[max_index]
        # print(cut_clique)

        cliques = split_by_continuity(cut_clique)

        exit_vector = []

        for atom_idx in cut_clique:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in cut_clique:
                    exit_vector.append(neighbor_idx)

        if len(cliques) == 1:
            for i in range(len(exit_vector)):
                if exit_vector[i] > max(cut_clique):
                    exit_vector[i] = exit_vector[i] - len(cut_clique)

        else:
            for i in range(len(exit_vector)):
                ind = find_indices(cliques, exit_vector[i])
                # print(ind)
                # print(exit_vector[i])
                if ind == 0:
                    exit_vector[i] =  exit_vector[i] - len(cliques[ind])
                    # print(123)
                else:
                    length = 0
                    for i in range(ind + 1):
                        length+= len(cliques[i])
                    exit_vector[i] = exit_vector[i] - length


        if len(exit_vector) != 1:
            return [], [], []
        else:
            return mol, cut_clique, exit_vector
    except:
        print("Could not align")
        return [], [], []

        # # Get exit vector  和cut_clique
        # exit_vector = []
        # if len(cut_clique) <= 4:
        #     for atom_idx in cut_clique:
        #         if mol.GetAtomWithIdx(atom_idx).IsInRing():
        #             exit_vector.append(atom_idx)
        # else:
        #     for atom_idx in cut_clique:
        #         atom = mol.GetAtomWithIdx(atom_idx)
        #         neighbors = atom.GetNeighbors()
        #         for neighbor in neighbors:
        #             neighbor_idx = neighbor.GetIdx()
        #             if neighbor_idx not in cut_clique:
        #                 exit_vector.append(neighbor_idx)


        # print(exit_vector)
        # 从头开始编写frag RWmol()
        # frag_tree = FragTree(smi_molecule, cut_clique, exit_vector)
        # print(frag_tree.nodes[0].smiles)
        # for node in frag_tree.nodes:
        #         print(node.smiles)
        # smi_frag = decode(frag_tree)

        # return (aligned_mols[0], aligned_mols[1]), nodes_to_keep, exit_vectors


def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False


def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z


def to_graph_mol(mol, dataset):
    if mol is None:
        return [], []
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None, None

    # 移除立体化学信息， 比如楔形键(inward)和虚线键(outward)
    Chem.RemoveStereochemistry(mol)

    edges = []
    nodes = []

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_idx, end_idx = min(begin_idx, end_idx), max(begin_idx, end_idx)
        if mol.GetAtomWithIdx(begin_idx).GetAtomicNum() == 0 or mol.GetAtomWithIdx(end_idx).GetAtomicNum() == 0:
            continue
        else:
            edges.append((begin_idx, bond_dict[str(bond.GetBondType())], end_idx))
            if  bond_dict[str(bond.GetBondType())] == 3:
                return [], []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        valence = atom.GetTotalValence()
        charge = atom.GetFormalCharge()
        atom_str = "%s%i(%i)" % (symbol, valence, charge)
        # print(atom_str)

        if atom_str not in dataset_info(dataset)['atom_types']:
            # print(1)
            if "*" in atom_str:
                continue
            else:
                # print('unrecognized atom type %s' % atom_str)
                return [], []

        nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), len(dataset_info(dataset)['atom_types'])))

    return nodes, edges


def set_seed(seed, cuda=False):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print('set seed for random numpy and torch')


# def test_align_mol_to_frags(task_name='Mutagenicity'):
#
#     data_path = './data/Mutagenicity/Mutagenicity_train.csv'
#     data = pd.read_csv(data_path)
#     smiles_list = data['smiles'].tolist()
#     smiles = list(set(smiles_list))
#     indices_to_delete = []
#     # print(len(smiles))
#
#     for i, smi in enumerate(smiles):
#         # print(smi)
#         mol_out, clique, exit_points = align_mol_to_frags(smi, task_name=task_name)
#         if mol_out == []:
#             print(smi)
#             indices_to_delete.append(i)
#             print(indices_to_delete)
        # nodes_out, edges_out = to_graph_mol(mol_out, 'Mutagenicity')  # 输入分子
#
#         cliques = split_by_continuity(clique)
#         # print(cliques)
#         fragment_edges = []
#         for k in (edges_out):
#             start = k[0]
#             end = k[2]
#             edge = k[1]
#             if start in clique or end in clique:
#                 continue
#
#             if len(cliques) == 1:
#                 if start > max(clique):
#                     start = start - len(clique)
#                 if end > max(clique):
#                     end = end - len(clique)
#             else:
#                 s_ind = find_indices(cliques, start)
#                 e_ind = find_indices(cliques, start)
#
#                 if s_ind == 0:
#                     start = start - len(cliques[s_ind])
#                 else:
#                     len1 = 0
#                     for i in range(s_ind + 1):
#                         len1 += len(cliques[i])
#                     start = start - len1
#
#                 if e_ind == 0:
#                     end = end - len(cliques[s_ind])
#                 else:
#                     len2 = 0
#                     for i in range(e_ind + 1):
#                         len2 += len(cliques[i])
#                     start = start - len2
#
#             fragment_edges.append((start, edge, end))
#
#         if min(len(fragment_edges), len(edges_out)) <= 0:
#             indices_to_delete.append(i)
#     print(indices_to_delete)
#     data_cleaned = data.drop(indices_to_delete)
#     data_cleaned.to_csv('./data/Mutagenicity/Mutagenicity_train.csv', index=False)



if __name__ == '__main__':
    #
    # test_align_mol_to_frags()

    # smiles = ['Nc1c(C(=O)O)cc([N+](=O)[O-])c2c1C(=O)c1ccccc1C2=O', 'Cc1ccc(N)cc1[N+](=O)[O-]',
    #           'COc1cc([N+](=O)[O-])ccc1N',
    #           '[N-]=[N+]=Nc1ccc(F)c([N+](=O)[O-])c1', 'O=[N+]([O-])c1cc2ccccc2c2ccccc12',
    #           'O=[N+]([O-])c1ccc2c3ccccc3c3cccc4ccc1c2c43']

    smiles = ['O=[N+]([O-])c1cc2ccccc2c2ccccc12']
    for s in smiles:
        mol_out, clique, exit_points = align_mol_to_frags(s, task_name='Mutagenicity')
        # print(clique)
        # print(exit_points)
        # frag_smi = Chem.MolFragmentToSmiles(mol_out, clique, kekuleSmiles=True)
        # new_mol = Chem.MolFromSmiles(frag_smi, sanitize=False)
        # new_mol = copy_edit_mol(new_mol).GetMol()
        # new_mol = sanitize(new_mol)
        # frag_smi = Chem.MolToSmiles(new_mol)
        # print(frag_smi)







