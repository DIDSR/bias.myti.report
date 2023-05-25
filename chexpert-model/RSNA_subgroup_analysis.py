from math import inf, nan
import torch
from torch import nn, optim
from args import TestArgParser
from logger import Logger
from predict import Predictor, EnsemblePredictor
from saver import ModelSaver
from data import get_loader
from eval import Evaluator
from constants import *
# from scripts.get_cams import save_grad_cams
from dataset import TASK_SEQUENCES
from nuancedmetric import *
import argparse
import os
import pandas as pd
import sklearn.metrics as sk_metrics
import numpy as np
import json

custom_subgroups ={ 
    'sex':{'M','F'},
    'race':{'White', 'Black'},
    'COVID_positive':{'Yes', 'No'},
    'modality':{'CR', 'DX'}
}

def get_confusion_matrix(predictions, groundtruth, threshold):
    """
    function to compute confusion matrix
    """
    tp = np.sum(np.logical_and(predictions >= threshold, groundtruth == 1))
    tn = np.sum(np.logical_and(predictions < threshold, groundtruth == 0))
    fp = np.sum(np.logical_and(predictions >= threshold, groundtruth == 0))
    fn = np.sum(np.logical_and(predictions < threshold, groundtruth == 1))
    return tp, tn, fp, fn

def subgroup_calculation():
    """
    function to read scores and patient attribute info, and calculate subgroup bias measurements
    """
    sub_dir = args.sub_dir
    threshold = args.threshold
    subgroup_for_eval = []
    subgroup_for_eval.append(args.test_subgroup)
    prediction_file = args.prediction_file
    if args.post_processed == False:
      # read scores
      predictions = pd.read_csv(os.path.join(args.pred_dir, prediction_file))
      info_pred = predictions.copy()
    else:
      # read validation_2 patient list, and drop duplicate patient ids
      validation_2_list = pd.read_csv(args.info_file)
      validation_2_list.drop_duplicates(subset="patient_id", keep='first', inplace=True)
      # read scores
      predictions = pd.read_csv(os.path.join(args.pred_dir, prediction_file))
      
      # mapping patient labels to output score
      info_pred = predictions.copy()
      info_cols = ['F','M','Black', "White", "Yes", 'No']
      cols = [c for c in info_pred.columns if c not in ['patient_id']]
      info_pred = info_pred.rename(columns={c:f"{c} score" for c in cols})
      for c in info_cols:
          info_pred[c] = info_pred['patient_id'].map(validation_2_list.set_index("patient_id")[c])

    # subgroup calculations

    # nuanced auroc and AEG
    subgroup_df = pd.DataFrame().assign(sub_1=info_pred['F'], sub_2=info_pred['M'])
    true_label = pd.DataFrame().assign(target=info_pred['Yes'])
    pred_prob = pd.DataFrame().assign(target=info_pred['Yes score'])
    nuance = NuancedROC()
    temp = nuance.score(true_label, pred_prob, subgroup_df)
    aeg = AEG()
    temp2 = aeg.score(true_label, pred_prob, subgroup_df)
    output_nuance_auc_aeg = pd.concat([temp, temp2], join='outer',axis=1)
    output_nuance_auc_aeg.to_csv(os.path.join(sub_dir, f"subgroup_nuance_auc_{prediction_file}"))

    # fairness measurements
    dp = {}
    for sub in subgroup_for_eval:
        for grp in custom_subgroups[sub]:
            info_sub = info_pred[info_pred[grp].isin(['1'])]
            task_gt = info_sub["Yes"]
            task_pred = info_sub["Yes score"]
            dp[f"{grp}"] = {}
            tp, tn, fp, fn = get_confusion_matrix(task_pred, task_gt, threshold)
            #TODO: compute ratio or difference
            # Add random seeds info
            dp[f"{grp}"]['Rand_seed'] = f"Rand_{args.rand}"
            # Average Score
            dp[f"{grp}"]['Average Score'] = np.mean(task_pred)
            # Demographic Parity criteria
            dp[f"{grp}"]['Demographic Parity (thres)'] = (tp+fp) / (tp+tn+fp+fn)
            # Equalized Odds criteria (sensitivity)
            dp[f"{grp}"]['TPR'] = tp / (tp+fn)
            # Predictive Rate Parity
            dp[f"{grp}"]['PPV'] = tp / (tp+fp)
            # specificity
            dp[f"{grp}"]['TNR'] = tn / (tn+fp)
            # AUROC
            dp[f"{grp}"]['AUROC'] = sk_metrics.roc_auc_score(y_score=task_pred, y_true=task_gt)
            # Overall AUROC
            dp[f"{grp}"]['Overall AUROC'] = sk_metrics.roc_auc_score(y_score=info_pred["Yes score"], y_true=info_pred["Yes"])
            # NLL (uncertainty estimation)
            p = torch.tensor(info_pred["Yes score"])
            l = torch.tensor(info_pred["Yes"])
            nll_criterion = nn.BCELoss()
            dp[f"{grp}"]['NLL_overall'] = nll_criterion(p, l.double()).item()
            p_sub = torch.tensor(task_pred.reset_index(drop=True))
            l_sub = torch.tensor(task_gt.reset_index(drop=True))
            dp[f"{grp}"]['NLL'] = nll_criterion(p_sub, l_sub.double()).item()      
    
    # output the computed measurements      
    metrics_summary = pd.DataFrame.from_dict({i: dp[i] 
                           for i in dp.keys()},
                       orient='index')
    metrics_summary.to_csv(os.path.join(sub_dir, f"subgroup_fairness_{prediction_file}"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand',type=int)
    parser.add_argument('--sub_dir',type=str)
    parser.add_argument('--pred_dir',type=str)
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--test_subgroup',type=str)
    parser.add_argument('--prediction_file',type=str)
    parser.add_argument('--info_file',type=str)
    parser.add_argument('--post_processed',default=False,type=bool)
    args = parser.parse_args()
    subgroup_calculation()
    print("Done\n")
