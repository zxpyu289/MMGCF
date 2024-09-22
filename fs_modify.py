import os
import sys
import copy
import time
import torch
import argparse
import numpy as np
import pickle as pkl
import environment as env
import torch.nn.functional as F
# for linux env.
sys.path.insert(0,'..')

from tqdm import tqdm
from rdkit import Chem
from math import sqrt
from util.chemutils import *
from data_loader import CondDataset
from torch_geometric.data import Data
from util.similary import get_similarity
from train import Trainer, read_molecules
from torch.utils.data import DataLoader
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rgcn.data_preprocess import load_graph
from util.mol_tree import MolTree, MolTreeNode
from rgcn.data_preprocess import construct_rgcn_data
from rgcn.rgcn import EarlyStopping,RGCN, collate_molgraphs, pos_weight
from data_utils import to_graph_mol, find_indices, split_by_continuity, atom_types, set_seed, add_atoms, bond_dict
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def get_parser():
    parser = argparse.ArgumentParser()
    # ******data args******
    parser.add_argument('--path', type=str, default='./data_preprocessed/HIV/', help='path of dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size.')
    parser.add_argument('--edge_unroll', type=int, default=12, help='max edge to model for each node in bfs order.')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle data for each epoch')
    parser.add_argument('--num_workers', type=int, default=10, help='num works to generate data.')


    # ******model args******
    parser.add_argument('--name', type=str, default='base',
                        help='model name, crucial for test and checkpoint initialization')
    parser.add_argument('--deq_type', type=str, default='random', help='dequantization methods.')
    parser.add_argument('--deq_coeff', type=float, default=0.9,
                        help='dequantization coefficient.(only for deq_type random)')
    parser.add_argument('--num_flow_layer', type=int, default=12,
                        help='num of affine transformation layer in each timestep')
    parser.add_argument('--gcn_layer', type=int, default=3, help='num of rgcn layers')
    # TODO: Disentangle num of hidden units for gcn layer, st net layer.
    parser.add_argument('--nhid', type=int, default=128, help='num of hidden units of gcn')
    parser.add_argument('--nout', type=int, default=128, help='num of out units of gcn')

    parser.add_argument('--st_type', type=str, default='exp',
                        help='architecture of st net, choice: [sigmoid, exp, softplus, spine]')

    # ******for sigmoid st net only ******
    parser.add_argument('--sigmoid_shift', type=float, default=2.0, help='sigmoid shift on s.')

    # ******optimization args******
    parser.add_argument('--all_save_prefix', type=str, default='./', help='path of save prefix')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--learn_prior', action='store_true', default=False, help='learn log-var of gaussian prior.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--warm_up', action='store_true', default=True,
                        help='Add warm-up and decay to the learning rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--is_bn', action='store_true', default=True, help='batch norm on node embeddings.')
    parser.add_argument('--is_bn_before', action='store_true', default=False,
                        help='batch norm on node embeddings on st-net input.')
    parser.add_argument('--scale_weight_norm', action='store_true', default=False,
                        help='apply weight norm on scale factor.')
    parser.add_argument('--divide_loss', action='store_true', default=False, help='divide loss by length of latent.')

    parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('--init_checkpoint', type=str, default='./good_ckpt/checkpoint306',
                        help='initialize from a checkpoint, if None, do not restore')
    # parser.add_argument('--init_checkpoint', type=str, default='./save_pretrain/train/checkpoint29',
    #                     help='initialize from a checkpoint, if None, do not restore')
    # parser.add_argument('--init_checkpoint', type=str, default='./save_pretrain/BBBP/checkpoint29',
    #                     help='initialize from a checkpoint, if None, do not restore')

    # parser.add_argument('--init_checkpoint', type=str, default='./save_pretrain/ESOL/checkpoint330',
    #                     help='initialize from a checkpoint, if None, do not restore')

    # ******generation args******
    parser.add_argument('--temperature', type=float, default=0.75, help='temperature for normal distribution')
    # parser.add_argument('--gen_num', type=int, default=10,
    #                     help='num of molecules to generate on each call to train.generate')
    parser.add_argument('--len_freedom_x', type=int, default=0, help='the minimum adjust of generated length')
    parser.add_argument('--len_freedom_y', type=int, default=0, help='the maximum adjust of generated length')

    # ******demo args******
    parser.add_argument('--max_atoms', type=int, default=89, help='maximum atoms of generated mol')
    parser.add_argument('--atom_types', type=int, default=14, help='number of atom types in generated mol')
    parser.add_argument('--edge_types', type=int, default=4, help='number of edge types in generated mol')


    parser.add_argument('--patience', type=int, default=30, help='the number of epochs to be tolerated')
    parser.add_argument('--task_name', type=str, default='HIV', help='task name of the dataset')
    parser.add_argument('--mode', type=str, default='higher', help='directions for optimising assessment indicators')
    parser.add_argument('--gpu', type=int, default=0, help='GPU Id to use')
    parser.add_argument('--bin_path', type=str, default='./log/HIV_logs/rgcn_data/dgl_graph.bin')
    parser.add_argument('--group_path', type=str, default='./log/HIV_logs/rgcn_data/HIV_group.csv')

    return parser

parser = get_parser()
args = parser.parse_args()


args.cuda = not args.no_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)

# 加载数据
# 需要被生成的片段， 生成长度， 在生成过程中保持不变的片段
full_nodes, full_edges, full_smi, data_config = read_molecules(args.path)
train_dataloader = DataLoader(CondDataset(full_nodes, full_edges, full_smi),
                              batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              num_workers=args.num_workers)


trainer = Trainer(train_dataloader, data_config, args, all_train_smiles=full_smi)

checkpoint = torch.load(args.init_checkpoint)
trainer._model.load_state_dict(checkpoint['model_state_dict'], strict=False)
trainer._model.eval()
temperature = args.temperature


# print(args)

if args.gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 加载数据
with open('./rgcn/result/hyperparameter_{}.pkl'.format(args.task_name), 'rb') as f:
    hyperparameter = pkl.load(f)

rgcn_hidden_feats = hyperparameter['rgcn_hidden_feats']
ffn_hidden_feats = hyperparameter['ffn_hidden_feats']
classification = hyperparameter['classification']

classifier = RGCN(ffn_hidden_feats=ffn_hidden_feats,
                  ffn_dropout=0,
                  rgcn_node_feats=40,
                  rgcn_hidden_feats=rgcn_hidden_feats,
                  rgcn_drop_out=0,
                  num_output=1,
                  classification=classification)
seed = 0

stopper = EarlyStopping(patience=args.patience, task_name=args.task_name + '_' + str(seed + 1),
                        mode=args.mode, filename='./rgcn/model/{}_3_early_stop.pth'.format(args.task_name))
classifier = classifier.to(device)
stopper.load_checkpoint(classifier)
classifier.eval()

train_set, val_set, test_set, task_number = load_graph(
    bin_path=args.bin_path,
    group_path=args.group_path,
    classification=classification,
    seed=2024,
    random_shuffle=False)


num2bond = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE}
num2bond_symbol = {0: '=', 1: '==', 2: '==='}
num2symbol = {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O', 10: 'O', 11: 'S',
                  12: 'S', 13: 'S'}
maximum_valence = {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4, 13: 6, 14: 3}


def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x, y, 1) )
        dfs(stack, y, x.idx)
        # stack.append( (y, x, 0))

def bfs(stack, x, fa):
    stop_node = MolTreeNode("")
    stop_node.idx = -2
    current_layer = [[x, fa]]
    next_layer = []
    while len(current_layer) > 0:
        for active_node in current_layer:
            x = active_node[0]
            fa = active_node[1]
            for y in x.neighbors:
                if y.idx == fa.idx:
                    continue
                stack.append((x, y, 1))
                # stack.append((y, x, 0))
                next_layer.append([y, x])
                #[node in the next layer, its parent node]
            # stack.append((x, stop_node, 2))
        # 2 denotes stop nodes
        current_layer = next_layer
        next_layer = []

def get_trace(smile):

    tree = MolTree(smile)
    super_root = MolTreeNode("")
    super_root.idx = -1

    # print(smile)
    for i, node in enumerate(tree.nodes):
        node.idx = i
    stack = []
    # dfs(stack, tree.nodes[0], -1)
    bfs(stack, tree.nodes[0], super_root)

    trace = [tree.nodes[0]]  # [Cl c1ccccc1 CC c1ccccc1]
    for x, y, d in stack:
        trace.append(y)

    return trace


def largest_mol(smiles):
    ss = smiles.split(".")
    ss.sort(key=lambda a: len(a))
    return ss[-1]


def get_model_output(smile):

    graph = construct_rgcn_data(smile, smask=[])
    graph = graph.to(device)
    node_feats = graph.ndata['node'].float().to(device)
    edge_feats = graph.edata['edge'].long().to(device)
    smask_feats = graph.ndata['smask'].unsqueeze(dim=1).float().to(device)
    org_prob = classifier(graph, node_feats, edge_feats, smask_feats)[0]
    # y_pred = torch.sigmoid(org_prob)
    # y_pred = round(y_pred.item(), 2)
    return org_prob

def get_frag_edges_adj(clique, edges_out):

    cliques = split_by_continuity(clique)

    fragment_edges = []
    for i in (edges_out):
        start = i[0]
        end = i[2]
        edge = i[1]

        if start in clique or end in clique:
            continue

        if len(cliques) == 1:
            if start > max(clique):
                start = start - len(clique)
            if end > max(clique):
                end = end - len(clique)
        else:
            if start > min(clique):
                s_ind = find_indices(cliques, start)
                if s_ind == 0:
                    start = start - len(cliques[s_ind])
                else:
                    len1 = 0
                    for i in range(s_ind + 1):
                        len1 += len(cliques[i])
                    start = start - len1

            if end > min(clique):
                e_ind = find_indices(cliques, end)
                # print("end_1:", end)

                if e_ind == 0:
                    end = end - len(cliques[e_ind])
                else:
                    len2 = 0
                    for i in range(e_ind + 1):
                        len2 += len(cliques[i])
                    end = end - len2

        fragment_edges.append((start, edge, end))


    frag_edge = torch.zeros([4, 89, 89])  # (4, 89, 89)
    for i in (fragment_edges):
        start = i[0]
        end = i[2]
        edge = i[1]
        frag_edge[edge, start, end] = 1.0
        frag_edge[edge, end, start] = 1.0


    frag_edge = frag_edge.detach().numpy()

    return frag_edge

def edge_to_convert_adj(edges_out):

    full_edge = torch.zeros([args.edge_types, args.max_atoms, args.max_atoms])  # (4, 89, 89)
    for i in (edges_out):
        start = i[0]
        end = i[2]
        edge = i[1]
        full_edge[edge, start, end] = 1.0
        full_edge[edge, end, start] = 1.0

    full_edge = full_edge.detach().numpy()

    return full_edge


def node_to_convert_adj(nodes_out):
    full_node = torch.zeros([89, 14])  # (89, 14)
    for i in range(len(nodes_out)):
        for j in range(len(nodes_out[0])):
            full_node[i][j] = nodes_out[i][j]

    full_node = full_node.detach().numpy()

    return full_node


# 无操作的函数
def no_operation(nodes, edges, tr):
    atom_symbol = []
    bond = []

    cur_clique = tr.clique

    for i, atom in enumerate(nodes):
        if i not in cur_clique: continue
        if max(atom) > 0:
            atom_symbol.append((i, np.argmax(atom)))

    for i in (edges):
        start = i[0]
        end = i[2]
        edge = i[1]
        if start in cur_clique and end in cur_clique:
            bond.append((start, edge, end))
    return atom_symbol, bond


# 删除的函数
def remove_leaf():
    atom_symbol = []
    bond = []
    return atom_symbol, bond


# 重新生成的函数
def generate_substructure(tr, full_node, full_edge, link):

    len_freedom = 0
    prior_node_dist = torch.distributions.normal.Normal(torch.zeros([trainer._model.node_dim]).cuda(),
                                                        temperature * torch.ones([trainer._model.node_dim]).cuda())
    prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([trainer._model.bond_dim]).cuda(),
                                                        temperature * torch.ones([trainer._model.bond_dim]).cuda())

    atom_symbol = []
    bond = []
    exit_vector = []
    for item in link:
        exit_vector.append(item[0])
        exit_vector.append(item[2])

    cur_clique = tr.clique

    valences = []

    cur_node_features = torch.tensor(full_node).unsqueeze(0).cuda()
    cur_adj_features = torch.tensor(full_edge).unsqueeze(0).cuda()

    sub_mol = Chem.RWMol()
    # new_nodes = [i for i in cur_clique] + [i for i in range(len(cur_clique), (len(cur_clique) + len_freedom))]
    new_nodes = [i for i in range(len(cur_clique))] + [i for i in range(len(cur_clique), (len(cur_clique) + len_freedom))]

    cur_node_features_save = cur_node_features.clone()
    cur_adj_features_save = cur_adj_features.clone()
    valences_save = copy.deepcopy(valences)

    for i, new_node in enumerate(new_nodes):
        atom_id = cur_clique[new_node]
        flag = True
        while flag:
            flag = False
            max_x = 0
            while max_x < 1:
                latent_node = prior_node_dist.sample().view(1, -1)  # (1, 14)
                latent_node = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                      latent_node, mode=0).view(-1)  # (14, )
                max_x = max(latent_node)

            feature_id = torch.argmax(latent_node).item()
            cur_node_features[0, new_node, feature_id] = 1.0
            add_atoms(sub_mol, [feature_id])

            new_edges = [i for i, x in enumerate(valences) if x > 0]
            if atom_id in exit_vector:
                valences.append(maximum_valence[feature_id] - 1)
            else:
                valences.append(maximum_valence[feature_id])

            # then generate edges
            max_y = 0
            bond_num = []
            while len(new_edges) > 0:  #
                edge_p = []
                for new_edge in new_edges:
                    latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                    latent_edge = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                          latent_edge,
                                                                          mode=1, edge_index=torch.Tensor([[new_edge, new_node]]).long().cuda()).view(-1)[:3]
                    for a, x in enumerate(latent_edge):
                        if a >= valences[new_edge] or a >= valences[new_node] or x < 0:
                            latent_edge[a] *= 0
                    edge_p.append(latent_edge)
                max_y = max([max(e) for e in edge_p])
                # print(max_y)
                index = torch.argmax(torch.tensor([max(e) for e in edge_p])).item()
                # edge_discrete_id = torch.argmax(torch.tensor(edge_p[index])).item()
                edge_discrete_id = torch.argmax(edge_p[index].clone().detach()).item()
                if max_y < 1 and len(bond_num) > 0:
                    break

                chosen_edge = new_edges[index]

                bond.append([cur_clique[new_node], edge_discrete_id, cur_clique[chosen_edge]])

                valences[new_node] -= (edge_discrete_id + 1)
                valences[chosen_edge] -= (edge_discrete_id + 1)
                new_edges.remove(chosen_edge)
                sub_mol.AddBond(int(new_node), int(chosen_edge), num2bond[edge_discrete_id])
                bond_num.append([int(new_node), int(chosen_edge)])

                cur_adj_features[0, edge_discrete_id, new_node, chosen_edge] = 1.0
                cur_adj_features[0, edge_discrete_id, chosen_edge, new_node] = 1.0

            if [i for i,x in enumerate(valences) if x > 0]==[] :
                flag=True
                for b in bond_num:
                    sub_mol.RemoveBond(b[0],b[1])
                sub_mol.RemoveAtom(int(new_node))
                valences = copy.deepcopy(valences_save)
                cur_node_features = cur_node_features_save.clone()
                cur_adj_features = cur_adj_features_save.clone()
            else:
                atom_symbol.append((cur_clique[new_node], feature_id))
                valences_save = copy.deepcopy(valences)
                cur_node_features_save = cur_node_features.clone()
                cur_adj_features_save = cur_adj_features.clone()

    # check submol

    mol = sub_mol.GetMol()
    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
    sub_smi = Chem.MolToSmiles(final_mol, isomericSmiles=False)
    if '.' in sub_smi or Chem.MolFromSmiles(sub_smi) == None:
        return None, None
    else:
        return atom_symbol, bond


# 对环操作的函数
def modify_ring(tr, nodes, edges):
    prior_edge_dist = torch.distributions.normal.Normal(torch.zeros([trainer._model.bond_dim]).cuda(),
                                                        temperature * torch.ones([trainer._model.bond_dim]).cuda())

    clique = tr.clique

    ori_ring_key = [item for item in edges if item[0] in clique and item[2] in clique]

    atom_symbol = []
    bond = []

    node_symbol = []
    for i, atom in enumerate(nodes):
        # if i not in clique: continue
        if max(atom) > 0:
            node_symbol.append(np.argmax(atom))

    ring_mol = Chem.RWMol()

    for i_c, number in enumerate(node_symbol):
        if i_c not in clique: continue
        new_atom = Chem.Atom(num2symbol[number])
        charge_num = int(atom_types[number].split('(')[1].strip(')'))
        new_atom.SetFormalCharge(charge_num)
        ring_mol.AddAtom(new_atom)


    valences = [maximum_valence[s] for s in node_symbol]
    for i in range(len(node_symbol)):
            valences[i] = 1

    all_nodes = node_to_convert_adj(nodes)
    all_edges = edge_to_convert_adj(edges)

    cur_node_features = torch.tensor(all_nodes).unsqueeze(0).cuda()
    cur_adj_features = torch.tensor(all_edges).unsqueeze(0).cuda()

    #  new_edge, new_node
    gen_ring_key = [[item[0], item[2], -1] for item in ori_ring_key]
    # new_nodes = clique
    new_nodes = [i for i in range(len(clique))]
    # new_ring = []
    node_map = {clique[i]: new_nodes[i] for i in range(len(clique))}

    # 使用node_map将gen_ring_key中的节点替换为new_nodes中的索引
    converted_gen_ring_key = [[node_map[pair[0]], node_map[pair[1]], pair[2]] for pair in gen_ring_key]


    for new_node in new_nodes:
        atom_id = clique[new_node]
        feature_id = node_symbol[atom_id]
        atom_symbol.append((atom_id, feature_id))
        # add_atoms(ring_mol, [feature_id])

        # get the dst
        # new_edges = [item[1] for item in gen_ring_key if item[0] == clique[new_node] and item[2] == -1]  # and valences[item[1]] > 0]
        new_edges = [item[1] for item in converted_gen_ring_key if
                     item[0] == new_node and item[2] == -1]
        max_y = 0
        bond_num = []
        while len(new_edges) > 0:  #
            edge_p = []

            for new_edge in new_edges:
                latent_edge = prior_edge_dist.sample().view(1, -1)  # (1, 4)
                latent_edge = trainer._model.flow_core.module.reverse(cur_node_features, cur_adj_features,
                                                                      latent_edge,
                                                                      mode=1, edge_index=torch.Tensor([[new_edge, new_node]]).long().cuda()).view(-1)[:3]

                for a, x in enumerate(latent_edge):
                    if a >= valences[new_edge] or a >= valences[atom_id] or x < 0:
                        latent_edge[a] *= 0
                edge_p.append(latent_edge)
            max_y = max([max(e) for e in edge_p])
            index = torch.argmax(torch.tensor([max(e) for e in edge_p])).item()
            # edge_discrete_id = torch.argmax(edge_p[index].clone().detach()).item()
            edge_discrete_id = torch.argmax(edge_p[index].clone().detach()).item()

            chosen_edge = new_edges[index]
            # traj.append([new_node, chosen_edge, edge_discrete_id])
            valences[new_node] -= (edge_discrete_id + 1)
            valences[chosen_edge] -= (edge_discrete_id + 1)
            new_edges.remove(chosen_edge)
            ring_mol.AddBond(int(new_node), int(chosen_edge), num2bond[edge_discrete_id])
            bond_num.append([int(new_node), int(chosen_edge)])
            # bond.append([new_node, edge_discrete_id, chosen_edge])

            # record the generate key
            for i, item in enumerate(gen_ring_key):
                if converted_gen_ring_key[i][0] == new_node and converted_gen_ring_key[i][1] == chosen_edge:
                    converted_gen_ring_key[i][2] = edge_discrete_id

            # valid = env.check_valency(rw_mol)
            cur_adj_features[0, edge_discrete_id, new_node, chosen_edge] = 1.0
            cur_adj_features[0, edge_discrete_id, chosen_edge, new_node] = 1.0
            bond.append((clique[new_node], edge_discrete_id, clique[chosen_edge]))

    mol = ring_mol.GetMol()
    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
    ring_smi = Chem.MolToSmiles(final_mol, isomericSmiles=False)
    if '.' in ring_smi or Chem.MolFromSmiles(ring_smi) == None:
        return None
    else:
        return atom_symbol, bond


def get_recon_sub(smile, tr, link):

    # 对叶节点有三种选择：0 无操作 1 删除 2 重新生成
    # 对非叶节点主要是环 0 无操作 1 重新生成

    start_mol = get_mol(smile)
    ssr_lst = [list(x) for x in Chem.GetSymmSSSR(start_mol)]

    # [Cl c1ccccc1 CC c1ccccc1]
    nodes, edges = to_graph_mol(start_mol, 'Mutagenicity')

    full_node_adj = node_to_convert_adj(nodes)
    full_edge_adj = edge_to_convert_adj(edges)

    if tr.is_leaf and tr.clique not in ssr_lst:
        choice_ls = [0, 1, 2]
        random_choice = np.random.choice(choice_ls, 1)[0]
        if random_choice == 0:
            recon_atom, recon_edge = no_operation(nodes, edges, tr)
        elif random_choice == 1:
            recon_atom, recon_edge = remove_leaf()
        elif random_choice == 2:
            recon_atom, recon_edge = generate_substructure(tr, full_node_adj, full_edge_adj, link)
        else:
            raise Exception('Invalid Operation ')
        # print('is_leaf')
    elif tr.clique in ssr_lst:
        # print('is_ring')
        choice_ls = [0, 1]
        random_choice = np.random.choice(choice_ls, 1)[0]
        if random_choice == 0:
            recon_atom, recon_edge = no_operation(nodes, edges, tr)
        elif random_choice == 1:
            recon_atom, recon_edge = modify_ring(tr, nodes, edges)
        else:
            raise Exception('Invalid Operation ')
    else:
        # print('no')
        choice_ls = [0]
        random_choice = np.random.choice(choice_ls, 1)[0]
        if random_choice == 0:
            recon_atom, recon_edge = no_operation(nodes, edges, tr)
        else:
            raise Exception('Invalid Operation ')

    return recon_atom, recon_edge


def reorganize(s, node,bond, del_clique, trace):
    mol = get_mol(s)
    for tr in trace:
        del_fa = []
        if tr.clique in del_clique: continue
        for y in tr.neighbors:
            if y.clique in del_clique:
                del_fa = y.clique
        for atom_idx in tr.clique:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            for nei in neighbors:
                neighbor_idx = nei.GetIdx()
                if neighbor_idx not in tr.clique and neighbor_idx not in del_fa:

                    edge = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                    if edge:
                        begin_idx, end_idx = min(atom_idx, neighbor_idx), max(atom_idx, neighbor_idx)
                        b = (begin_idx, bond_dict[str(edge.GetBondType())], end_idx)
                        if b not in bond:
                            bond.append((begin_idx, bond_dict[str(edge.GetBondType())], end_idx))

    unique_combinations = set()
    unique_bonds = []

    for item in bond:
        begin = item[0]
        end = item[2]

        # 确保 (begin, end) 组合只出现一次
        # 使用元组 (min(begin, end), max(begin, end)) 保持一致性
        combination = (min(begin, end), max(begin, end))

        if combination not in unique_combinations:
            unique_combinations.add(combination)
            unique_bonds.append(item)
    atom = sorted(node, key=lambda item: item[0])

    return unique_bonds, atom

def get_motif_link(s, trace):
    link = []
    mol = get_mol(s)
    for tr in trace:

        for atom_idx in tr.clique:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbors = atom.GetNeighbors()
            for nei in neighbors:
                neighbor_idx = nei.GetIdx()
                if neighbor_idx not in tr.clique:
                    edge = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                    if edge:
                        begin_idx, end_idx = min(atom_idx, neighbor_idx), max(atom_idx, neighbor_idx)
                        b = (begin_idx, bond_dict[str(edge.GetBondType())], end_idx)
                        if b not in link:
                            link.append((begin_idx, bond_dict[str(edge.GetBondType())], end_idx))
    return link


def dfs_recon(smile):

    trace = get_trace(smile)

    node_symbol = []
    bond = []
    del_clique = []
    link = get_motif_link(smile, trace)

    for tr in trace:
        flag = True
        while flag:
            recon_sub_atom, recon_sub_edge = get_recon_sub(smile, tr, link)
            if recon_sub_atom != None:
                flag = False
                if recon_sub_atom == []:
                    # 表示该叶节点的操作是删除
                    del_clique.append(tr.clique)
                for atom in recon_sub_atom:
                    if atom not in node_symbol:
                        node_symbol.append(atom)
                for b in recon_sub_edge:
                    if b not in bond:
                        bond.append(b)
    unique_bonds, atom_symbol = reorganize(smile, node_symbol, bond, del_clique, trace)

    clique = []
    for c in del_clique:
        clique.extend(c)

    if len(clique) > 0:
        frag_edges = get_frag_edges_adj(clique, unique_bonds)
    else:
        frag_edges = edge_to_convert_adj(unique_bonds)

    rw_mol = Chem.RWMol()
    for atom in atom_symbol:
        if atom[0] in clique: continue
        number = atom[1]
        new_atom = Chem.Atom(num2symbol[number])
        charge_num = int(atom_types[number].split('(')[1].strip(')'))
        new_atom.SetFormalCharge(charge_num)
        rw_mol.AddAtom(new_atom)

    for bond in range(3):  # (4,89,89)
        for start in range(89):
            for end in range(start + 1, 89):
                if frag_edges[bond][start][end] == 1:
                    rw_mol.AddBond(start, end, num2bond[bond])

    mol = rw_mol.GetMol()
    final_mol = env.convert_radical_electrons_to_hydrogens(mol)
    new_smi = Chem.MolToSmiles(final_mol, isomericSmiles=False)

    if '.' in new_smi or Chem.MolFromSmiles(new_smi) == None or new_smi == '':
        return None
    else:
        return new_smi


def _select_examples(cond, examples, nmols):
    result = []

    def cluster_score(e):
        score = cond(e) * e['sim_score']
        return score

    for e in examples:
        # check if actually is (since call could have been zero)
        if cluster_score(e):
            result.append(e)

    # sort by similarity
    result = sorted(result, key=lambda v: v['sim_score'], reverse=True)
    # back fill
    result.extend(sorted(examples, key=lambda v: v['sim_score'] * cond(v), reverse=True))

    final_result = result[:nmols]

    return list(filter(cond, final_result))


def find_counterf(examples, nmols: int = 1):

    def is_counter(e):
        return e['class'] != examples[0]['class']
        # print(e[0]['class'])

    result = _select_examples(is_counter, examples[1:], nmols)
    for r in result:
        r['label'] = 'Counterfactual'

    return examples[:1] + result

def find_rcf(examples, nmols: int = 1):

    delta: Union[Any, Tuple[float, float]] = (-1, 1),
    delta = 0.5

    if type(delta) is float:
        delta = (-delta, delta)

    def high_or_low(e):
        return (e['output'] + delta[0] >= examples[0]['output']) or (e['output'] + delta[1] <= examples[0]['output'])

    result = _select_examples(high_or_low, examples[1:], nmols)
    for r in result:
        r['label'] = 'Counterfactual'

    return examples[:1] + result


def save_result_smile(input_smi,smile_muted, label, ori_pred):
    smi = input_smi
    similarity_measure = "combined"
    similarity_measure_graph = "neural_encoding"

    original_molecule = construct_rgcn_data(smi, smask=[])
    similarity, make_encoding, original_encoding = get_similarity(similarity_measure,
                                                                  classifier,
                                                                  original_molecule,
                                                                  smi,
                                                                  fp_len=1024,
                                                                  fp_rad=2)

    graph_similarity, graph_make_encoding, graph_original_encoding = get_similarity(similarity_measure_graph,
                                                                                    classifier,
                                                                                    original_molecule,
                                                                                    smi,
                                                                                    fp_len=1024,
                                                                                    fp_rad=2)

    overall = []
    overall.append({
        'smiles': smi,
        'sim_score': 1.0,
        'graph_sim_score': 1.0,
        'type': 'bin_classification',
        'output': ori_pred,
        'for_explanation': label,
        'class': label.item(),
        'label':None

    })

    all_mols = [smi2mol(s) for s in smile_muted]
    all_canon = [largest_mol(mol2smi(m, canonical=True)) if m else None for m in all_mols]
    seen = set()
    to_keep = [False for _ in all_canon]
    for i in range(len(all_canon)):
        if all_canon[i] and all_canon[i] not in seen:
            to_keep[i] = True
            seen.add(all_canon[i])

    # now do filter
    filter_smiles = [s for i, s in enumerate(smile_muted) if to_keep[i]]

    sim_score = [similarity(make_encoding(construct_rgcn_data(smiles, smask=[]), smiles), original_encoding) for smiles
                 in smile_muted]

    graph_sim_score = [
        graph_similarity(graph_make_encoding(construct_rgcn_data(smiles, smask=[])), graph_original_encoding) for smiles
        in smile_muted]

    out_score = [get_model_output(s_m) for s_m in smile_muted]

    score = [torch.sigmoid(out_s) for out_s in out_score]
    # pred_score = [out_s.item() for out_s in out_score]
    # torch.sigmoid(org_prob)
    pred_score = [torch.sigmoid(out_s).item() for out_s in out_score]

    pred_class = torch.tensor([1.0 if p_s > 0.5 else 0.0 for p_s in pred_score])

    for i, (sm, pred, s, s_graph, o_s, p_c) in enumerate(zip(filter_smiles, pred_score,sim_score, graph_sim_score,score, pred_class)):
        cf_queue = ({
            'smiles':sm,
            'pred_score': pred,
            'sim_score': s,
            'graph_sim_score': s_graph,
            'type': 'bin_classification',
            'output': o_s,
            'for_explanation': p_c,
            'class': p_c.item(),
            'label': None
        })

        overall.append(cf_queue)

    return overall


def save_reg_result(input_smi,smile_muted, y_true):
    smi = input_smi
    similarity_measure = "combined"
    similarity_measure_graph = "neural_encoding"

    original_molecule = construct_rgcn_data(smi, smask=[])
    similarity, make_encoding, original_encoding = get_similarity(similarity_measure,
                                                                  classifier,
                                                                  original_molecule,
                                                                  smi,
                                                                  fp_len=1024,
                                                                  fp_rad=2)

    graph_similarity, graph_make_encoding, graph_original_encoding = get_similarity(similarity_measure_graph,
                                                                                    classifier,
                                                                                    original_molecule,
                                                                                    smi,
                                                                                    fp_len=1024,
                                                                                    fp_rad=2)

    overall = []
    overall.append({
        'smiles': smi,
        'sim_score': 1.0,
        'graph_sim_score': 1.0,
        'type': 'regression',
        'output': y_true,
        'label': None

    })

    all_mols = [smi2mol(s) for s in smile_muted]
    all_canon = [largest_mol(mol2smi(m, canonical=True)) if m else None for m in all_mols]
    seen = set()
    to_keep = [False for _ in all_canon]
    for i in range(len(all_canon)):
        if all_canon[i] and all_canon[i] not in seen:
            to_keep[i] = True
            seen.add(all_canon[i])

    # now do filter
    filter_smiles = [s for i, s in enumerate(smile_muted) if to_keep[i]]

    sim_score = [similarity(make_encoding(construct_rgcn_data(smiles, smask=[]), smiles), original_encoding) for smiles
                 in smile_muted]

    graph_sim_score = [
        graph_similarity(graph_make_encoding(construct_rgcn_data(smiles, smask=[])), graph_original_encoding) for smiles
        in smile_muted]

    out_score = [get_model_output(s_m) for s_m in smile_muted]

    for i, (sm, s, s_graph, o_s) in enumerate(zip(filter_smiles, sim_score, graph_sim_score, out_score)):
        cf_queue = ({
            'smiles':sm,
            'sim_score': s,
            'graph_sim_score': s_graph,
            'type': 'regression',
            'output': o_s.item(),
            'label': None

        })

        overall.append(cf_queue)

    return overall


def _generate_cf(smile, label, ori_pred):
    num_random_samples = 100

    result_lst = []
    randomized_smile = [smile for _ in range(num_random_samples)]
    for smi in randomized_smile:
        # print('-----------------------------------')
        # print(smi)
        smi_cf = dfs_recon(smi)
        if smi_cf != None and smi_cf not in result_lst:
            result_lst.append(smi_cf)

    if classification:
        result = save_result_smile(smile, result_lst, label, ori_pred)
    else:
        result =  save_reg_result(smile, result_lst, ori_pred)

    return result


def evaluate_validity(data):
    y_eq_list = []

    for item in data:
        y_pred = item[0]['output']
        y_pred_label = torch.round(y_pred)

        cf_pred = item[1]['output']
        cf_pred_label = torch.round(cf_pred)

        y_eq = torch.where(y_pred_label == cf_pred_label, torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
        y_eq_list.append(y_eq)

    y_eq_tensor = torch.cat(y_eq_list)

    valid = torch.mean(y_eq_tensor)
    return valid.item()


def evaluate_proximity(data):

    proxi_list = []
    proxi_graph_list = []

    for item in data:
        if item[1]['label'] != 'Counterfactual':
            continue

        pr = item[1]['sim_score']
        pr_graph = item[1]['graph_sim_score']

        proxi_list.append(pr)
        proxi_graph_list.append(pr_graph)

    proxi_result = np.mean(proxi_list)
    proxi_graph_result = np.mean(proxi_graph_list)

    return proxi_result, proxi_graph_result

def compute_reg_valid(data):
    count = 0

    delta: Union[Any, Tuple[float, float]] = (-1, 1),
    delta = 0.5
    if type(delta) is float:
        delta = (-delta, delta)

    def is_high(cf_pred, orig_pred):
        return cf_pred + delta[0] >= orig_pred

    def is_low(cf_pred, orig_pred):
        return cf_pred + delta[1] <= orig_pred

    for item in data:
        y_pred = item[0]['output']

        cf_pred = item[1]['output']

        if is_high(cf_pred, y_pred):
            count += 1
        elif is_low(cf_pred, y_pred):
            count += 1
        else:
            count = count + 0
    valid = count / len(data)

    return valid

def evaluate(data, metrics):
    score_valid = 0.0
    proxi = 0.0
    proxi_graph = 0.0
    if 'validity' in metrics:
        score_valid = evaluate_validity(data)

    if 'proximity' or 'proximity_graph' in metric:
        proxi, proxi_graph = evaluate_proximity(data)

    if 'reg_validity' in metrics:
        score_valid = compute_reg_valid(data)

    return score_valid, proxi, proxi_graph


def main_for_class(path, seed):
    num = 200
    full = []
    s_time = time.time()
    sum = 0
    cont = 0
    for data_item in tqdm(test_set[:num]):
        # cont += 1
        # if cont < 161:
        #     continue
        smile, g_rgcn, label, smask, sub_name = data_item
        g_rgcn = g_rgcn.to(device)

        node_feats = g_rgcn.ndata['node'].float().to(device)
        edge_feats = g_rgcn.edata['edge'].long().to(device)
        smask_feats = g_rgcn.ndata['smask'].unsqueeze(dim=1).float().to(device)

        org_prob = classifier(g_rgcn, node_feats, edge_feats, smask_feats)[0]
        y_pred = torch.sigmoid(org_prob)
        pred_label = torch.round(y_pred)

        ori_label = label.item()
        if pred_label.item() == 1 and ori_label == 1:
            sum += 1
            result_space =  _generate_cf(smile, label, y_pred)
            cf_result = find_counterf(result_space)
            if (len(cf_result) == 1):
                cf_result.append(cf_result[0])
            else:
                full.append(cf_result)

    e_time = time.time()
    run_t = round(((e_time - s_time) / sum), 3)
    metrics = ['proximity', 'validity', 'proximity_graph']
    score_valid, proxi, proxi_graph = evaluate(full, metrics)

    file_path = os.path.join(path, '{}_{}'.format(args.task_name, seed + 1))
    with open(file_path, 'a') as f:
        for line in full:
            f.write('%s %s %s %s %s \n' % (
            line[0]['smiles'], line[0]['output'], line[1]['smiles'], line[1]['output'], line[1]['sim_score']))
    return score_valid, proxi, proxi_graph, run_t



def main_for_reg(path, seed):
    train_set, val_set, test_set, task_number = load_graph(
        bin_path=args.bin_path,
        group_path=args.group_path,
        classification=classification,
        seed=2024,
        random_shuffle=False)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             collate_fn=collate_molgraphs)


    start_time = time.time()
    cont = 0

    full = []
    for data_item in tqdm(test_loader):
        cont += 1

        smiles, g_rgcn, labels, smask, sub_name = data_item

        smile = smiles[0]

        labels = labels.unsqueeze(dim=1).float().to(device)
        y_true = round(labels.squeeze().item(), 3)

        result_space = _generate_cf(smile, label=labels, ori_pred=y_true)

        cf_result = find_rcf(result_space)

        if (len(cf_result) == 1):
            cf_result.append(cf_result[0])
        else:
            full.append(cf_result)


    e_time = time.time()
    run_t = round(((e_time - start_time) / cont), 3)
    metrics = ['proximity', 'reg_validity', 'proximity_graph']
    score_valid, proxi, proxi_graph = evaluate(full, metrics)

    file_path = os.path.join(path, '{}_{}'.format(args.task_name, seed + 1))
    with open(file_path, 'a') as f:
        for line in full:
            f.write('%s %s %s %s %s \n' % (
            line[0]['smiles'], line[0]['output'], line[1]['smiles'], line[1]['output'], line[1]['sim_score']))
    return score_valid, proxi, proxi_graph, run_t



def test():
    smi = "CN(C)CCCN=c1c2ccccc2[nH]c2ccc([N+](=O)[O-])cc12"
    num_random_samples = 100
    smiles = randomized_smile = [smi for _ in range(num_random_samples)]
    for s in smiles:
        tree = MolTree(s)
        # for node in tree.nodes:
        #     # print(node.smiles, [x.smiles for x in node.neighbors])
        #     print(node.smiles, [x.smiles for x in node.neighbors])
        # print('--------------------------------------------------------')
        for i, node in enumerate(tree.nodes):
            node.idx = i

        stack = []
        dfs(stack, tree.nodes[0], -1)
        trace = [tree.nodes[0]]
        for x, y, d in stack:
            trace.append(y)
            # print(x.smiles, y.smiles, d)
        # print('------------------------------')
        #
        link = get_motif_link(s, trace)
        node_symbol = []
        bond = []

        del_clique = []
        for tr in trace:
            flag = True
            while flag:
                recon_sub_atom, recon_sub_edge = get_recon_sub(s, tr, link)
                if recon_sub_atom != None:
                    flag = False
                    if recon_sub_atom == []:
                        # 表示该叶节点的操作是删除
                        del_clique.append(tr.clique)
                    for atom in recon_sub_atom:
                        if atom not in node_symbol:
                            node_symbol.append(atom)
                    for b in recon_sub_edge:
                        if b not in bond:
                            bond.append(b)


        # 记录motif-tree中motif和motif的连接情况
        mol = get_mol(s)
        for tr in trace:
            del_fa = []
            if tr.clique in del_clique : continue
            for y in tr.neighbors:
                if y.clique in del_clique:
                    del_fa = y.clique
            for atom_idx in tr.clique:
                atom = mol.GetAtomWithIdx(atom_idx)
                neighbors = atom.GetNeighbors()
                for nei in neighbors:
                    neighbor_idx = nei.GetIdx()
                    if neighbor_idx not in tr.clique and neighbor_idx not in del_fa:

                        edge = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                        if edge:
                            begin_idx, end_idx = min(atom_idx, neighbor_idx), max(atom_idx, neighbor_idx)
                            b = (begin_idx,  bond_dict[str(edge.GetBondType())], end_idx)
                            if b not in bond:
                                bond.append((begin_idx,  bond_dict[str(edge.GetBondType())], end_idx))

        unique_combinations = set()
        unique_bonds = []

        for item in bond:
            begin = item[0]
            end = item[2]

            # 确保 (begin, end) 组合只出现一次
            # 使用元组 (min(begin, end), max(begin, end)) 保持一致性
            combination = (min(begin, end), max(begin, end))

            if combination not in unique_combinations:
                unique_combinations.add(combination)
                unique_bonds.append(item)
        atom_symbol = sorted(node_symbol, key=lambda item: item[0])

        clique = []
        for c in del_clique:
            clique.extend(c)

        if len(clique) > 0:
            frag_edges = get_frag_edges_adj(clique, unique_bonds)
        else:
            frag_edges = edge_to_convert_adj(unique_bonds)


        rw_mol = Chem.RWMol()
        for atom in atom_symbol:
            if atom[0] in clique: continue
            number = atom[1]
            new_atom = Chem.Atom(num2symbol[number])
            charge_num = int(atom_types[number].split('(')[1].strip(')'))
            new_atom.SetFormalCharge(charge_num)
            rw_mol.AddAtom(new_atom)

        for bond in range(3):  # (4,89,89)
            for start in range(89):
                for end in range(start + 1, 89):
                    if frag_edges[bond][start][end] == 1:
                        rw_mol.AddBond(start, end, num2bond[bond])

        mol = rw_mol.GetMol()
        final_mol = env.convert_radical_electrons_to_hydrogens(mol)
        new_smi = Chem.MolToSmiles(final_mol, isomericSmiles=False)
        print(new_smi)


if __name__ == '__main__':
    # test()

    eval_time = []
    eval_valid = []
    eval_proximity = []
    eval_proximity_Graph = []
    base_path = './build_mol_2/'

    for i in range(10):
        print('***************************************************************************************************')

        valid, proxi, proxi_graph, run_tim = main_for_class(base_path, i)
        # valid, proxi, proxi_graph, run_tim = main_for_reg(base_path, i)
        print(
            '{}, seed {}, validity {}, proximity {}, proximity on graph {}, average time {}'.format('mmgcf', i + 1,
                                                                                                    valid, proxi,
                                                                                                    proxi_graph,
                                                                                                    run_tim))

        eval_time.append(run_tim)

        eval_valid.append(valid)
        eval_proximity.append(proxi)
        eval_proximity_Graph.append(proxi_graph)

    mean_valid = np.mean(eval_valid)
    variance_valid = np.var(eval_valid)

    mean_time = np.mean(eval_time)
    variance_time = np.var(eval_time)

    mean_proxi = np.mean(eval_proximity)
    variance_proxi = np.var(eval_proximity)

    mean_proxi_graph = np.mean(eval_proximity_Graph)
    variance_proxi_graph = np.var(eval_proximity_Graph)

    print(f"Mean of eval_valid: {mean_valid}")
    print(f"Variance of eval_valid: {variance_valid}")

    print(f"Mean of eval_time: {mean_time}")
    print(f"Variance of eval_time: {variance_time}")

    print(f"Mean of eval_proxi: {mean_proxi}")
    print(f"Variance of eval_proxi: {variance_proxi}")

    print(f"Mean of eval_proxi_graph: {mean_proxi_graph}")
    print(f"Variance of eval_proxi_graph: {variance_proxi_graph}")