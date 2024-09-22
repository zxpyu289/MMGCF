import itertools
import random
import os
import torch as th
import pandas as pd
import numpy as np
import numpy as np
import rdkit
import rdkit.Chem as Chem
import dgl
import sys
sys.path.insert(0,'..')

from rdkit.Chem import BRICS
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import RDConfig
from dgl import DGLGraph
from dgl.data.graph_serialize import save_graphs, load_graphs, load_labels
from util.chemutils import decomp, brics_decomp, get_mol


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format( x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom, use_chirality=True):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def etype_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i + 1
            # a = i

    bond_feats_2 = bond.GetIsConjugated()  #共轭系统通常涉及交替的单键和双键，这种结构有助于电子的离域化，增强了分子的稳定性。例如，在芳香环中，所有键通常是共轭的。
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    return index


def construct_rgcn_data(smiles, smask):
    # g = dgl.DGLGraph()

    # add nodes
    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()
    # g.add_nodes(num_atoms)
    atoms_feature_all = []
    smask_list = []
    for i, atom in enumerate(mol.GetAtoms()):
        atom_feature = atom_features(atom)
        atoms_feature_all.append(atom_feature)
        if i in smask:
            smask_list.append(0)
        else:
            smask_list.append(1)

    atoms_feature_all_array = np.array(atoms_feature_all)
    # g.ndata["node"] = th.tensor(atoms_feature_all_array)
    # g.ndata["smask"] = th.tensor(smask_list).float()

    # add edges
    src_list = []
    dst_list = []
    etype_feature_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        etype_feature = etype_features(bond)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
        etype_feature_all.append(etype_feature)
        etype_feature_all.append(etype_feature)

    g = dgl.graph((src_list, dst_list), num_nodes=num_atoms)

    # g.add_edges(src_list, dst_list)
    # g.edata["edge"] = th.tensor(etype_feature_all)
    g.ndata["node"] = th.tensor(atoms_feature_all_array)
    g.ndata["smask"] = th.tensor(smask_list).float()
    g.edata["edge"] = th.tensor(etype_feature_all)

    return g


def build_graph_data(dataset_smiles, labels_name, smiles_name):
    dataset_gnn = []
    failed_molecule = []
    labels = dataset_smiles[labels_name]
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    for i, smiles in enumerate(smilesList):
        # try:
        g_rgcn = construct_rgcn_data(smiles, smask=[])
        molecule = [smiles, g_rgcn, labels[i], split_index[i]]
        dataset_gnn.append(molecule)
        print('{}/{} molecule is transformed to mol graph! {} is transformed failed!'.format(i + 1, molecule_number,
                                                                                                 len(failed_molecule)))
        # except:
        #     print('{} is transformed to mol graph failed!'.format(smiles))
        #     molecule_number = molecule_number - 1
        #     failed_molecule.append(smiles)
    print('{}({}) is transformed to mol graph failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def build_motif_data(dataset_smiles,labels_name,smiles_name):
    dataset_gnn = []
    failed_molecule = []
    split_index = dataset_smiles['group']
    smilesList = dataset_smiles[smiles_name]
    molecule_number = len(smilesList)
    labels = dataset_smiles[labels_name]
    for i, smiles in enumerate(smilesList):
        # print(smiles)
        mol = get_mol(smiles)

        res = list(BRICS.FindBRICSBonds(mol))
        # print(res)
        if len(res) == 0:
            motif_dir, edges = decomp(mol)
        else:
            motif_dir, edges = brics_decomp(mol)

        substructure_mask = []

        for m in range(len(motif_dir)):
            substructure_mask.append(motif_dir[m])

        for j, motif_mask_i in enumerate(substructure_mask):
            try:
                g_rgcn = construct_rgcn_data(smiles, smask=motif_mask_i)
                molecule = [smiles, g_rgcn, labels.loc[i], split_index.loc[i], motif_mask_i]
                dataset_gnn.append(molecule)
                print('{}/{}, {}/{} molecule is transformed to mol graph! {} is transformed failed!'.format(j + 1,
                                                                                                        len(substructure_mask),
                                                                                                        i + 1,
                                                                                                        molecule_number,
                                                                                                        len(failed_molecule)))
            except:
                print('{} is transformed to mol graph failed!'.format(smiles))
                molecule_number = molecule_number - 1
                failed_molecule.append(smiles)
    print('{}({}) is transformed to mol graph failed!'.format(failed_molecule, len(failed_molecule)))
    return dataset_gnn


def load_graph(bin_path, group_path, smask_path=None,classification=True, random_shuffle=True, seed=2022):
    data = pd.read_csv(group_path)
    smiles = data.smiles.values
    group = data.group.to_list()
    # load substructure name
    if 'sub_name' in data.columns.tolist():
        sub_name = data['sub_name']
    else:
        sub_name = ['noname' for x in group]

    if random_shuffle:
        random.seed(seed)
        random.shuffle(group)
    homog, detailed_information = load_graphs(bin_path)
    labels = detailed_information['labels']

    # load smask
    if smask_path is None:
        smask = [-1 for x in range(len(group))]
    else:
        smask = np.load(smask_path, allow_pickle=True)

    # calculate not_use index
    train_index = []
    val_index = []
    test_index = []
    for index, group_index in enumerate(group):
        if group_index == 'training':
            train_index.append(index)
        if group_index == 'valid':
            val_index.append(index)
        if group_index == 'test':
            test_index.append(index)
    task_number = 1
    train_set = []
    val_set = []
    test_set = []
    for i in train_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i]]
        train_set.append(molecule)

    for i in val_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i]]
        val_set.append(molecule)

    for i in test_index:
        molecule = [smiles[i], homog[i], labels[i], smask[i], sub_name[i]]
        test_set.append(molecule)
    print(len(train_set), len(val_set), len(test_set), task_number)
    return train_set, val_set, test_set, task_number


if __name__ =='__main__':
    # tasks = ['Mutagenicity']
    # task_list = ['ESOL','BBBP', 'hERG']
    task_list = ['HIV']
    for task in task_list:
        input_csv = '../data/{}/'.format(task) + task + '.csv'
        output_graph_path = '../log/' + task + '_logs/rgcn_data/dgl_graph.bin'
        output_group_path = '../log/' + task + '_logs/rgcn_data/' + task + '_group.csv'

        output_motif_path = '../log/' + task + '_logs/rgcn_data/' + task + '_motif.bin'
        output_motif_group_path = '../log/' + task + '_logs/rgcn_data/' + task + '_motif_group.csv'
        output_motif_mask_path = '../log/' + task + '_logs/rgcn_data/' + task + '_motif_mask.npy'

        path = '../log/' + task + '_logs/rgcn_data/'

        if not os.path.exists(path):
            os.makedirs(path)
        data = pd.read_csv(input_csv, index_col=None)
        smiles_name = 'smiles'
        labels_name = task

        data_set_gnn = build_graph_data(dataset_smiles=data, labels_name=labels_name, smiles_name=smiles_name)
        smiles, g_rgcn, labels, split_index = map(list, zip(*data_set_gnn))
        graph_labels = {'labels': th.tensor(labels)}
        split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
        split_index_pd.smiles = smiles
        split_index_pd.group = split_index
        split_index_pd.to_csv(output_group_path, index=False, columns=None)
        print('Molecules graph is saved!')
        save_graphs(output_graph_path, g_rgcn, graph_labels)

        # build data for motif
        data_set_gnn_for_motif = build_motif_data(dataset_smiles=data, labels_name=labels_name, smiles_name=smiles_name)
        smiles, g_rgcn, labels, split_index, smask = map(list, zip(*data_set_gnn_for_motif))
        graph_labels = {'labels': th.tensor(labels)}
        split_index_pd = pd.DataFrame(columns=['smiles', 'group'])
        split_index_pd.smiles = smiles
        split_index_pd.group = split_index

        split_index_pd.to_csv(output_motif_group_path, index=False, columns=None)
        smask_np = np.array(smask, dtype=object)

        np.save(output_motif_mask_path, smask_np)
        print('Molecules graph for brics is saved!')
        save_graphs(output_motif_path, g_rgcn, graph_labels)


