import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
def pred2label(prob):
    if prob < 0.5:
        return 0
    else:
        return 1

for task in ['ESOL', 'Mutagenicity', 'hERG', 'BBBP', 'HIV']:
    for group in ['test']:
        data = pd.read_csv('./prediction/summary/{}_mol_prediction_summary.csv'.format(task))
        if task in ['ESOL']:
            data = data[data['group']==group]
            r2 = r2_score(data['label'], data['pred_mean'])
            print('{} {} rmse: {}'.format(task, group, r2))
            print(len(data))
        if task in ['Mutagenicity', 'hERG', 'BBBP', 'HIV']:
            data = data[data['group']==group]
            pred_label_list = [pred2label(prob) for prob in data['pred_mean'].tolist()]
            roc = roc_auc_score(data['label'], data['pred_mean'])
            print('{} {} roc_auc: {}'.format(task, group, roc))
            print(len(data))