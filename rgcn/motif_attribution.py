import numpy as np
import pandas as pd
import torch
import pickle as pkl
import time
import os

from torch.optim import Adam
from torch.utils.data import DataLoader
from rgcn import collate_molgraphs, EarlyStopping, train_model, eval_model, set_random_seed, RGCN, pos_weight
from data_preprocess import load_graph
from rdkit import Chem
from rdkit.Chem import BRICS


def predict_substructure(seed, task_name, rgcn_hidden_feats=[64, 64, 64], ffn_hidden_feats=128, lr=0.0003, classification=True, sub_type='motif'):

    args = {}
    args['device'] = "cuda"
    args['node_data_field'] = 'node'
    args['edge_data_field'] = 'edge'
    args['substructure_mask'] = 'smask'
    args['classification'] = classification
    # model parameter
    args['num_epochs'] = 500
    args['patience'] = 30
    args['batch_size'] = 128
    args['mode'] = 'higher'
    args['in_feats'] = 40
    args['rgcn_hidden_feats'] = rgcn_hidden_feats
    args['ffn_hidden_feats'] = ffn_hidden_feats
    args['rgcn_drop_out'] = 0
    args['ffn_drop_out'] = 0
    args['lr'] = lr
    args['loop'] = True
    # task name (model name)
    args['task_name'] = task_name  # change
    args['data_name'] = task_name  # change
    args['bin_path'] = '../log/' + args['data_name'] + '_logs/rgcn_data/{}_{}.bin'.format(args['data_name'], sub_type)
    args['group_path'] = '../log/' + args['data_name'] + '_logs/rgcn_data/{}_{}_group.csv'.format(args['data_name'], sub_type)
    args['smask_path'] ='../log/' + args['data_name'] + '_logs/rgcn_data/{}_{}_mask.npy'.format(args['data_name'], sub_type)
    args['seed'] = seed

    print('***************************************************************************************************')
    print('{}, seed {}, substructure type {}'.format(args['task_name'], args['seed'] + 1, sub_type))
    print('***************************************************************************************************')
    train_set, val_set, test_set, task_number = load_graph(
        bin_path=args['bin_path'],
        group_path=args['group_path'],
        smask_path=args['smask_path'],
        classification=args['classification'],
        random_shuffle=False
    )
    print("Molecule graph is loaded!")
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args['batch_size'],
                              collate_fn=collate_molgraphs)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            collate_fn=collate_molgraphs)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs)
    if args['classification']:
        pos_weight_np = pos_weight(train_set)
        loss_criterion = torch.nn.BCEWithLogitsLoss(reduction='none',
                                                    pos_weight=pos_weight_np.to(args['device']))
    else:
        loss_criterion = torch.nn.MSELoss(reduction='none')

    model = RGCN(ffn_hidden_feats=args['ffn_hidden_feats'],
                 ffn_dropout=args['ffn_drop_out'],
                 rgcn_node_feats=args['in_feats'], rgcn_hidden_feats=args['rgcn_hidden_feats'],
                 rgcn_drop_out=args['rgcn_drop_out'],
                 num_output=1,
                 classification=args['classification'])

    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'] + '_' + str(seed + 1),
                            mode=args['mode'])
    model.to(args['device'])
    stopper.load_checkpoint(model)
    pred_name = '{}_{}_{}'.format(args['task_name'], sub_type, seed + 1)
    stop_test_list, _ = eval_model(args, model, test_loader, loss_criterion,
                                          out_path='../rgcn/prediction/{}/{}_test'.format(sub_type, pred_name))
    stop_train_list, _ = eval_model(args, model, train_loader, loss_criterion,
                                           out_path='../rgcn/prediction/{}/{}_train'.format(sub_type, pred_name))
    stop_val_list, _ = eval_model(args, model, val_loader, loss_criterion,
                                         out_path='../rgcn/prediction/{}/{}_val'.format(sub_type, pred_name))
    print('Mask prediction is generated!')

def prediction_summary(task_name, sub_type):
    print('{} {} sum succeed.'.format(task_name, sub_type))
    # 将训练集，验证集，测试集数据合并
    result_summary = pd.DataFrame()
    for i in range(3):
        seed = i + 1
        result_train = pd.read_csv('../rgcn/prediction/{}/{}_{}_{}_train_prediction.csv'.format(sub_type, task_name, sub_type, seed))
        result_val = pd.read_csv('../rgcn/prediction/{}/{}_{}_{}_val_prediction.csv'.format(sub_type, task_name, sub_type, seed))
        result_test = pd.read_csv('../rgcn/prediction/{}/{}_{}_{}_test_prediction.csv'.format(sub_type, task_name, sub_type, seed))

        group_list = ['training' for x in range(len(result_train))] + ['val' for x in range(len(result_val))] + ['test' for x in range(len(result_test))]
        result = pd.concat([result_train, result_val, result_test], axis=0)

        # mol是模型最初预测的时候给的结果，batch是会随机乱序的，所以需要重新排序
        result['group'] = group_list
        if sub_type == 'mol':
            result.sort_values(by='smiles', inplace=True)
        # 合并五个随机种子结果，并统计方差和均值
        if seed == 1:
            result_summary['smiles'] = result['smiles']
            result_summary['label'] = result['label']
            result_summary['sub_name'] = result['sub_name']
            result_summary['group'] = result['group']
            result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
        if seed > 1:
            result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
    pred_columnms = ['pred_{}'.format(i + 1) for i in range(3)]
    data_pred = result_summary[pred_columnms]
    result_summary['pred_mean'] = data_pred.mean(axis=1)
    result_summary['pred_std'] = data_pred.std(axis=1)
    dirs = '../rgcn/prediction/summary/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    result_summary.to_csv('../rgcn/prediction/summary/{}_{}_prediction_summary.csv'.format(task_name, sub_type), index=False)

def calculate_attribution(task_name, sub_type):
    attribution_result = pd.DataFrame()
    print('{} {}'.format(task_name, sub_type))
    result_sub = pd.read_csv('../rgcn/prediction/summary/{}_{}_prediction_summary.csv'.format(task_name, sub_type))
    result_mol = pd.read_csv('../rgcn/prediction//summary/{}_{}_prediction_summary.csv'.format(task_name, 'mol'))

    mol_pred_mean_list_for_sub = [result_mol[result_mol['smiles'] == smi]['pred_mean'].tolist()[0] for smi in result_sub['smiles'].tolist()]
    mol_pred_std_list_for_sub = [result_mol[result_mol['smiles'] == smi]['pred_std'].tolist()[0] for smi in result_sub['smiles'].tolist()]

    attribution_result['smiles'] = result_sub['smiles']
    attribution_result['label'] = result_sub['label']
    attribution_result['sub_name'] = result_sub['sub_name']
    attribution_result['group'] = result_sub['group']
    attribution_result['sub_pred_mean'] = result_sub['pred_mean']
    attribution_result['sub_pred_std'] = result_sub['pred_std']
    attribution_result['mol_pred_mean'] = mol_pred_mean_list_for_sub
    attribution_result['mol_pred_std'] = mol_pred_std_list_for_sub
    sub_pred_std_list = result_sub['pred_std']

    attribution_result['attribution'] = attribution_result['mol_pred_mean'] - attribution_result['sub_pred_mean']
    attribution_result['attribution_normalized'] = (np.exp(attribution_result['attribution'].values) - np.exp(
        -attribution_result['attribution'].values)) / (np.exp(attribution_result['attribution'].values) + np.exp(
        -attribution_result['attribution'].values))
    # attribution_result['attribution_normalized'] = 1 / (1 + np.exp(-2 * attribution_result['attribution'].values))


    dirs = '../rgcn/prediction/attribution/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    attribution_result.to_csv('../rgcn/prediction/attribution/{}_{}_attribution_negpos_summary.csv'.format(task_name, sub_type), index=False)


# def match_fragment_attribution(smiles, data):
#
#     brics_leaf = brics_leaf_structure(smiles)
#     mol = Chem.MolFromSmiles(smiles)
#     atom_num = mol.GetNumAtoms()
#     brics_leaf_sorted_id = sorted(range(len(brics_leaf['substructure'].keys())),
#                                             key=lambda k: list(brics_leaf['substructure'].keys())[k],
#                                             reverse=False)
#     frags_attribution = data[data['smiles'] == smiles].attribution_normalized.tolist()[atom_num:]
#     frags_pred =  data[data['smiles'] == smiles].sub_pred_mean.tolist()[atom_num:]
#     m2 = BRICS.BreakBRICSBonds(mol)
#     frags = Chem.GetMolFrags(m2, asMols=True)
#     frags_smi = [Chem.MolToSmiles(x, True) for x in frags]
#     sorted_frags_smi = [i for _, i in sorted(zip(list(brics_leaf_sorted_id), frags_smi), reverse=False)]
#     if len(sorted_frags_smi) != len(frags_attribution) or len(sorted_frags_smi) != len(frags_pred):
#         sorted_frags_smi = []
#         frags_attribution = []
#         frags_pred = []
#     return sorted_frags_smi, frags_attribution, frags_pred


def mol_pred_summary(task_name):

    result_summary = pd.DataFrame()
    for i in range(3):
        seed = i + 1
        result_train = pd.read_csv('../rgcn/prediction/mol/{}_mol_{}_train_prediction.csv'.format(task_name, seed))
        result_val = pd.read_csv('../rgcn/prediction/mol/{}_mol_{}_val_prediction.csv'.format(task_name, seed))
        result_test = pd.read_csv('../rgcn/prediction/mol/{}_mol_{}_test_prediction.csv'.format(task_name, seed))
        group_list = ['training' for x in range(len(result_train))] + ['val' for x in range(len(result_val))] + [ 'test' for x in range(len(result_test))]

        # 合并五个随机种子结果，并统计方差和均值
        result = pd.concat([result_train, result_val, result_test], axis=0)
        # mol是模型最初预测的时候给的结果，batch是会随机乱序的，所以需要重新排序
        result['group'] = group_list

        # 合并随机种子结果，并统计方差和均值
        if seed == 1:
            result_summary['smiles'] = result['smiles']
            result_summary['label'] = result['label']
            result_summary['sub_name'] = result['sub_name']
            result_summary['group'] = result['group']
            result_summary['pred_{}'.format(seed)] = result['pred'].tolist()
        if seed > 1:
            result_summary['pred_{}'.format(seed)] = result['pred'].tolist()

    pred_columnms = ['pred_{}'.format(i + 1) for i in range(3)]
    data_pred = result_summary[pred_columnms]
    result_summary['pred_mean'] = data_pred.mean(axis=1)
    result_summary['pred_std'] = data_pred.std(axis=1)
    dirs = '../rgcn/prediction/summary/'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    result_summary.to_csv('../rgcn/prediction/summary/{}_mol_prediction_summary.csv'.format(task_name),index=False)


if __name__ == '__main__':

    tasks = ['HIV']
    # tasks = ['ESOL', 'BBBP', 'hERG']
    for task in tasks:
        mol_pred_summary(task)
        for sub_type in ['motif']:
            with open('../rgcn/result/hyperparameter_{}.pkl'.format(task), 'rb') as f:
                hyperparameter = pkl.load(f)
            for i in range(3):
                predict_substructure(seed=i, task_name=task,
                                     rgcn_hidden_feats=hyperparameter['rgcn_hidden_feats'],
                                     ffn_hidden_feats=hyperparameter['ffn_hidden_feats'],
                                     lr=hyperparameter['lr'], classification=hyperparameter['classification'],
                                     sub_type=sub_type)
            prediction_summary(task, sub_type)
            calculate_attribution(task, sub_type)





