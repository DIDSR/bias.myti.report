from math import inf, nan
import torch

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
    tp = np.sum(np.logical_and(predictions >= threshold, groundtruth == 1))
    tn = np.sum(np.logical_and(predictions < threshold, groundtruth == 0))
    fp = np.sum(np.logical_and(predictions >= threshold, groundtruth == 0))
    fn = np.sum(np.logical_and(predictions < threshold, groundtruth == 1))
    return tp, tn, fp, fn

def subgroup_calculation():
    main_dir = args.test_dir
    rand = args.rand
    threshold = args.threshold
    subgroup_for_eval = []
    subgroup_for_eval.append(args.test_subgroup)
    # read validation_2 patient list, and drop duplicate patient ids
    validation_2_list = pd.read_csv(os.path.join(main_dir, "validation_2.csv"))
    validation_2_list.drop_duplicates(subset="patient_id", keep='first', inplace=True)
    groundtruth = pd.read_csv(os.path.join(main_dir, f"RAND_{rand}/output_model/CheXpert-Mimic_Resnet_subgroup_size_decay_90_lr_1e-5_exponential/results/all_by_patient_predictions.csv"))
    predictions = pd.read_csv(os.path.join(main_dir, f"RAND_{rand}/output_model/CheXpert-Mimic_Resnet_subgroup_size_decay_90_lr_1e-5_exponential/results/all_by_patient_predictions.csv"))
    # mapping patient labels to output score
    info_pred = predictions.copy()
    info_cols = ['F','M','Black', "White", "Yes", 'No']
    cols = [c for c in info_pred.columns if c not in ['patient_id']]
    info_pred = info_pred.rename(columns={c:f"{c} score" for c in cols})
    # info_grdt = groundtruth.copy()
    for c in info_cols:
        info_pred[c] = info_pred['patient_id'].map(validation_2_list.set_index("patient_id")[c])
    #   info_grdt[c] = info_grdt['patient_id'].map(validation_2_list.set_index("patient_id")[c])
    # print(info_pred.head(8))

    # subgroup calculations

    # nuanced auroc and AEG
    #TODO: get rid of hard code
    subgroup_df = pd.DataFrame().assign(sub_1=info_pred['White'], sub_2=info_pred['Black'])
    true_label = pd.DataFrame().assign(target=info_pred['Yes'])
    pred_prob = pd.DataFrame().assign(target=info_pred['Yes score'])
    nuance = NuancedROC()
    temp = nuance.score(true_label, pred_prob, subgroup_df)
    aeg = AEG()
    temp2 = aeg.score(true_label, pred_prob, subgroup_df)
    output_nuance_auc_aeg = pd.concat([temp, temp2], join='outer',axis=1)
    output_nuance_auc_aeg.to_csv(os.path.join(main_dir, f"RAND_{rand}/output_model/CheXpert-Mimic_Resnet_subgroup_size_decay_90_lr_1e-5_exponential/subgroup_nuance_auc.csv"))

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
            dp[f"{grp}"]['Rand_seed'] = f"Rand_{rand}"
            # Demographic Parity criteria
            dp[f"{grp}"]['Demographic Parity'] = (tp+fp) / (tp+tn+fp+fn)
            # Equalized Odds criteria
            dp[f"{grp}"]['TPR'] = tp / (tp+fn)
            # Predictive Rate Parity
            dp[f"{grp}"]['PPV'] = tp / (tp+fp)
            # AUROC
            dp[f"{grp}"]['AUROC'] = sk_metrics.roc_auc_score(y_score=task_pred, y_true=task_gt)

            
            # metrics = metric_calculation(task_pred, task_gt, threshold)

    metrics_summary = pd.DataFrame.from_dict({i: dp[i] 
                           for i in dp.keys()},
                       orient='index')
    metrics_summary.to_csv(os.path.join(main_dir, f"RAND_{rand}/output_model/CheXpert-Mimic_Resnet_subgroup_size_decay_90_lr_1e-5_exponential/subgroup_fairness.csv"))





    # don't use rows with missing values (-1)
    #task_gt = groundtruth[groundtruth[task] >= 0]
    #task_pred = predictions[groundtruth[task] >= 0]
    # print(task_gt[task].head(5))
    # print(task_pred[task].head(5))
    #if len(task_gt) <= 1:
    #    continue
    # get overall AUROC
    #if sum(task_gt[task]) == 0:
        # AUROC_dict[f'{task} (overall)'] = "none in gt"
    #    AUROC_dict[f'{task} (overall)'] = nan
    #elif sum(task_gt[task]) == len(groundtruth):
        # AUROC_dict[f'{task} (overall)'] = "all gt"
    #    AUROC_dict[f'{task} (overall)'] = inf
    #else:
    #    AUROC_dict[f'{task} (overall)'] = sk_metrics.roc_auc_score(y_true=task_gt[task], y_score=task_pred[task])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand',type=int,default=0)
    parser.add_argument('--test_dir',type=str)
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--test_subgroup',type=str)
    args = parser.parse_args()
    subgroup_calculation()
    print("Done\n")
