import rdkit
import os
import numpy as np
import rdkit.Chem as Chem
import pandas as pd
from rdkit.Chem import BRICS
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import RDConfig
from rdkit.Chem import FragmentCatalog
from rdkit.Chem import rdMMPA
from rdkit.Chem import rdMolAlign
from rdkit.Chem import Draw


MST_MAX_WEIGHT = 100
MAX_NCAND = 2000


def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def get_smiles(mol):

    return Chem.MolToSmiles(mol, kekuleSmiles=False)
    # if mol is not None:
    #     return Chem.MolToSmiles(mol, kekuleSmiles=False)
    # else:
    #     return ""


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    # get the fragment of clique
    Chem.Kekulize(mol)
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    Chem.SanitizeMol(mol)
    return new_mol


def brics_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) == 0:
        return [list(range(n_atoms))], []
    else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # break bonds between rings and non-ring atoms
    # print(cliques)
    clique_remove = []
    for c in cliques:
        # print("1:", c)
        if len(c) > 1:
            # print("2:", c)
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                clique_remove.append(c)
                cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                clique_remove.append(c)
                cliques.append([c[0]])
                breaks.append(c)

    cliques = [c for c in cliques if c not in clique_remove]

    # select atoms at intersections as motif
    for atom in mol.GetAtoms():
        if len(atom.GetNeighbors()) > 3 and not atom.IsInRing():
            cliques.append([atom.GetIdx()])
            for nei in atom.GetNeighbors():
                if [nei.GetIdx(), atom.GetIdx()] in cliques:
                    cliques.remove([nei.GetIdx(), atom.GetIdx()])
                    breaks.append([nei.GetIdx(), atom.GetIdx()])
                elif [atom.GetIdx(), nei.GetIdx()] in cliques:
                    cliques.remove([atom.GetIdx(), nei.GetIdx()])
                    breaks.append([atom.GetIdx(), nei.GetIdx()])
                cliques.append([nei.GetIdx()])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if len(c) > 0]

    # edges
    edges = []
    for bond in res:
        for c in range(len(cliques)):
            if bond[0][0] in cliques[c]:
                c1 = c
            if bond[0][1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))
    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges.append((c1, c2))

    return cliques, edges


def decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], []

    cliques = []  # 在环上的原子：一个环作为一个list存储其中的原子索引；不在环上的原子：[start, end]键开头结尾原子索引为一个list
    for bond in mol.GetBonds():  # 遍历键
        a1 = bond.GetBeginAtom().GetIdx()  # 键开头的原子索引
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():  # 如果键不在环中
            cliques.append([a1, a2])

    # get rings
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]  # Chem.GetSymmSSSR(mol):查看最小环
    cliques.extend(ssr)

    nei_list = [[] for i in range(n_atoms)]  # 对于每一个原子，存储所在cliques中的list的索引
    for i in range(len(cliques)):  # cliques中的每一个list内原子相互是邻居
        for atom in cliques[i]:
            nei_list[atom].append(i)

    # Merge Rings with intersection > 2 atoms  合并环
    for j in range(len(cliques)):
        if len(cliques[j]) <= 2: continue  # 选择cliques中 >2个原子的list
        for atom in cliques[j]:  # 遍历该list中原子
            for k in nei_list[atom]:  # 遍历该原子所在的所有cliques中的list
                if j >= k or len(cliques[k]) <= 2: continue  # 如果cliques存在第i个list之前有第j个list也是原子数>2
                inter = set(cliques[j]) & set(cliques[k])
                # if len(inter) >= 2:
                if len(inter) > 2:
                    cliques[j].extend(cliques[k])
                    cliques[j] = list(set(cliques[j]))  # 将第i和j内的原子去重都放在第i个（靠后）list中
                    cliques[k] = []  # 第j个list清空


    cliques = [c for c in cliques if len(c) > 0]  # 清除合并后为空的list
    # print(cliques)

    clique_remove = []
    breaks = []
    for c in cliques:
        # print("1:", c)
        if len(c) > 1:
            # print("2:", c)
            if mol.GetAtomWithIdx(c[0]).IsInRing() and not mol.GetAtomWithIdx(c[1]).IsInRing():
                clique_remove.append(c)
                if [c[1]] not in cliques:
                    cliques.append([c[1]])
                breaks.append(c)
            if mol.GetAtomWithIdx(c[1]).IsInRing() and not mol.GetAtomWithIdx(c[0]).IsInRing():
                clique_remove.append(c)
                if [c[0]] not in cliques:
                    cliques.append([c[0]])
                breaks.append(c)
    cliques = [c for c in cliques if c not in clique_remove]


    new_cliques = []
    for i_c in range(len(cliques)):
        if len(cliques[i_c]) == 2 :
            new_cliques.append(cliques[i_c])


    for n_c in range(len(new_cliques) - 1):
        if n_c >= len(new_cliques): break
        for k_c in range(n_c + 1, len(new_cliques)):
            if k_c >= len(new_cliques): break
            if len(set(new_cliques[n_c]) & set(new_cliques[k_c])) > 0:
                new_cliques[n_c] = list(set(new_cliques[n_c]) | set(new_cliques[k_c]))
                new_cliques[k_c] = []

        new_cliques = [n_c for n_c in new_cliques if len(n_c) > 0]

    new_cliques = [c for c in new_cliques if n_atoms > len(c) > 0]

    for new_clique in new_cliques:
        cliques = [c for c in cliques if not set(c).issubset(set(new_clique))]
    # print(cliques)

    cliques.extend(new_cliques)


    atom_order = {atom: index for index, atom in enumerate(sorted(set(sum(cliques, []))))}
    cliques = sorted(cliques, key=lambda clique: [atom_order[atom] for atom in clique])


    nei_list = [[] for i in range(n_atoms)]
    for m in range(len(cliques)):
        for atom in cliques[m]:
            nei_list[atom].append(m)


    # Build edges and add singleton cliques
    edges = defaultdict(int)
    for atom in range(n_atoms):
        if len(nei_list[atom]) < 1:
            continue

        cnei = nei_list[atom]
        bonds = [c for c in cnei if len(cliques[c]) == 2]  # bond的索引
        rings = [c for c in cnei if len(cliques[c]) > 4]   # ring的索引
        # len(ckiques) == 4

        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2):
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  # Multiple (n>2) complex rings
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei:
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1

        for i in range(len(cnei)):
            for j in range(i + 1, len(cnei)):
                c1, c2 = cnei[i], cnei[j]
                inter = set(cliques[c1]) & set(cliques[c2])
                if edges[(c1, c2)] < len(inter):
                    edges[(c1, c2)] = len(inter)  # cnei[i]


    for bond in breaks:
        for c in range(len(cliques)):
            if bond[0] in cliques[c]:
                c1 = c
            if bond[1] in cliques[c]:
                c2 = c
        edges[(c1, c2)] = 1
    #
    # for b in break_bond:
    #     for c in range(len(cliques)):
    #         if b[0] in cliques[c]:
    #             c1 = c
    #         if b[1] in cliques[c]:
    #             c2 = c
    #     edges[(c1, c2)] = 1

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]

    # print(edges)
    if len(edges) == 0:
        return cliques, edges


    # Compute Maximum Spanning Tree
    row, col, data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    return cliques, edges



if __name__ == "__main__":
    import sys
    from mol_tree import MolTree

    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)


    smiles = ["Cl.Cn1cc(N)cc1C(=O)Nc1cc(C(=O)Nc2ccc3cc(S(=O)(=O)O)cc(S(=O)(=O)O)c3c2)n(C)n1.[KH]"]
    # task_name = 'Mutagenicity'

    molecules = [get_mol(s) for s in smiles]

    def tree_test():


        for s in smiles:
            # print('-------------------------------------------')
            # print(s)

            tree = MolTree(s)
            for node in tree.nodes:
                # print(node.smiles, [x.smiles for x in node.neighbors])
                print(node.clique, [x.clique for x in node.neighbors])
                # if node.is_leaf:
                #     print(node.smiles)

    # tree_test()
    def test_decomp():
        for mol in molecules:

            res = list(BRICS.FindBRICSBonds(mol))
            print(res)
            if len(res) == 0:
                cliques, edges = decomp(mol)
            else:
                cliques,edges = brics_decomp(mol)
            print(cliques)
            print(edges)
            # for i in range(len(cliques)):
            #     smiles = Chem.MolFragmentToSmiles(mol, cliques[i], kekuleSmiles=True)
            #     new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
            #     new_mol = copy_edit_mol(new_mol).GetMol()
            #     new_mol = sanitize(new_mol)  # We assume this is not None
            #     Chem.SanitizeMol(mol)
            #     smile = Chem.MolToSmiles(new_mol)
            #     print(smile)
    test_decomp()




