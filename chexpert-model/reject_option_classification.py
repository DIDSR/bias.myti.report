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
    'sex':['M','F'],
    'race':['Black', 'White'],
    'COVID_positive':['Yes', 'No'],
    'modality':['CR', 'DX']
}

def get_confusion_matrix(predictions, groundtruth, threshold):
    tp = np.sum(np.logical_and(predictions >= threshold, groundtruth == 1))
    tn = np.sum(np.logical_and(predictions < threshold, groundtruth == 0))
    fp = np.sum(np.logical_and(predictions >= threshold, groundtruth == 0))
    fn = np.sum(np.logical_and(predictions < threshold, groundtruth == 1))
    return tp, tn, fp, fn

def ROC_mitigation():
    sub_dir = args.sub_dir
    threshold = args.threshold
    subgroup_for_eval = []
    subgroup_for_eval.append(args.test_subgroup)
    prediction_file = args.prediction_file
    if args.post_processed == False:
      predictions = pd.read_csv(os.path.join(args.pred_dir, prediction_file))
      info_pred = predictions.copy()
    else:
      # read validation_2 patient list, and drop duplicate patient ids
      validation_2_list = pd.read_csv(args.info_file)
      validation_2_list.drop_duplicates(subset="patient_id", keep='first', inplace=True)
      predictions = pd.read_csv(os.path.join(args.pred_dir, prediction_file))
      
      # mapping patient labels to output score
      info_pred = predictions.copy()
      info_cols = ['F','M','Black', "White", "Yes", 'No']
      cols = [c for c in info_pred.columns if c not in ['patient_id']]
      info_pred = info_pred.rename(columns={c:f"{c} score" for c in cols})
      for c in info_cols:
          info_pred[c] = info_pred['patient_id'].map(validation_2_list.set_index("patient_id")[c])
          
    # determine the privileged group
    dp = {}     
    test_list = custom_subgroups.get(args.test_subgroup)
    for grp in test_list:
        info_sub = info_pred[info_pred[grp].isin(['1'])]
        task_gt = info_sub["Yes"]
        task_pred = info_sub["Yes score"]
        dp[f"{grp}"] = {}
        dp[f"{grp}"]['Average Score'] = np.mean(task_pred)
    dp["Overall"] = {}    
    dp["Overall"]['Average Score'] = np.mean(info_pred["Yes score"])   
    
    if dp[test_list[0]]['Average Score'] > dp[test_list[1]]['Average Score']:
        group_p = test_list[0]
        group_u = test_list[1]
    else:
        group_p = test_list[1]
        group_u = test_list[0]
    
    info_sub_p = info_pred[info_pred[group_p].isin(['1'])]
    task_gt_p = info_sub_p["Yes"]
    task_pred_p = info_sub_p["Yes score"]
    info_sub_u = info_pred[info_pred[group_u].isin(['1'])]
    task_gt_u = info_sub_u["Yes"]
    task_pred_u = info_sub_u["Yes score"]    
    for threds in np.linspace(0, 0.4, 41):
        # privileged group
        tp_p, tn_p, fp_p, fn_p = get_confusion_matrix(task_pred_p, task_gt_p, threshold=threshold+threds)
        dp[f"{group_p}"][f"TPR_t_{threds}"] = tp_p / (tp_p+fn_p)
        # unprivileged group
        tp_u, tn_u, fp_u, fn_u = get_confusion_matrix(task_pred_u, task_gt_u, threshold=threshold-threds)
        dp[f"{group_u}"][f"TPR_t_{threds}"] = tp_u / (tp_u+fn_u)
        #Overall
        dp["Overall"][f"TPR_t_{threds}"] = (tp_u+tp_p) / (tp_p+fn_p+tp_u+fn_u)
            
    metrics_summary = pd.DataFrame.from_dict({i: dp[i] 
                           for i in dp.keys()},
                       orient='index')
    metrics_summary.to_csv(os.path.join(sub_dir, f"ROC_mitigation.csv"))


def ROC_mitigation_apply():
    # train on validation to find optimal thresholds
    sub_dir = args.sub_dir
    threshold = args.threshold
    subgroup_for_eval = []
    subgroup_for_eval.append(args.test_subgroup)
    prediction_file = args.prediction_file
    if args.post_processed == False:
      predictions = pd.read_csv(os.path.join(args.pred_dir, prediction_file))
      info_pred = predictions.copy()
    else:
      # read validation_2 patient list, and drop duplicate patient ids
      validation_2_list = pd.read_csv(args.info_file)
      validation_2_list.drop_duplicates(subset="patient_id", keep='first', inplace=True)
      predictions = pd.read_csv(os.path.join(args.pred_dir, prediction_file))
      
      # mapping patient labels to output score
      info_pred = predictions.copy()
      info_cols = ['F','M','Black', "White", "Yes", 'No']
      cols = [c for c in info_pred.columns if c not in ['patient_id']]
      info_pred = info_pred.rename(columns={c:f"{c} score" for c in cols})
      for c in info_cols:
          info_pred[c] = info_pred['patient_id'].map(validation_2_list.set_index("patient_id")[c])
          
    # determine the privileged group
    dp = {}     
    test_list = custom_subgroups.get(args.test_subgroup)
    for grp in test_list:
        info_sub = info_pred[info_pred[grp].isin(['1'])]
        task_gt = info_sub["Yes"]
        task_pred = info_sub["Yes score"]
        dp[f"{grp}"] = {}
        dp[f"{grp}"]['Average Score'] = np.mean(task_pred)
    dp["Overall"] = {}    
    dp["Overall"]['Average Score'] = np.mean(info_pred["Yes score"])   
    
    if dp[test_list[0]]['Average Score'] > dp[test_list[1]]['Average Score']:
        group_p = test_list[0]
        group_u = test_list[1]
    else:
        group_p = test_list[1]
        group_u = test_list[0]
    
    info_sub_p = info_pred[info_pred[group_p].isin(['1'])]
    task_gt_p = info_sub_p["Yes"]
    task_pred_p = info_sub_p["Yes score"]
    info_sub_u = info_pred[info_pred[group_u].isin(['1'])]
    task_gt_u = info_sub_u["Yes"]
    task_pred_u = info_sub_u["Yes score"]
    optimal_threds = 0
    min_diff = 1
    # searching the optimal thresholds    
    for threds in np.linspace(0, 0.49, 50):
        # privileged group
        tp_p, tn_p, fp_p, fn_p = get_confusion_matrix(task_pred_p, task_gt_p, threshold=threshold+threds)
        sen_1 = tp_p / (tp_p+fn_p+1e-8)
        dp[f"{group_p}"][f"TPR_t_{threds}"] = sen_1
        # unprivileged group
        tp_u, tn_u, fp_u, fn_u = get_confusion_matrix(task_pred_u, task_gt_u, threshold=threshold-threds)
        sen_2 = tp_u / (tp_u+fn_u+1e-8)
        dp[f"{group_u}"][f"TPR_t_{threds}"] = sen_2
        #Overall
        dp["Overall"][f"TPR_t_{threds}"] = (tp_u+tp_p) / (tp_p+fn_p+tp_u+fn_u+1e-8)
        if abs(sen_1-sen_2) < min_diff:
            min_diff = abs(sen_1-sen_2)
            optimal_threds = threds
        else:
            pass    
            
    print("The optimal threshold is")
    print(optimal_threds)        
    metrics_summary = pd.DataFrame.from_dict({i: dp[i] 
                           for i in dp.keys()},
                       orient='index')
    metrics_summary.to_csv(os.path.join(sub_dir, f"ROC_mitigation_validation.csv"))
    
    
    # apply to testing
    process_file = args.process_file
    if args.post_processed == False:
      predictions = pd.read_csv(os.path.join(args.pred_dir, process_file))
      info_pred = predictions.copy()
    else:
      # read validation_2 patient list, and drop duplicate patient ids
      validation_2_list = pd.read_csv(args.info_file_2)
      validation_2_list.drop_duplicates(subset="patient_id", keep='first', inplace=True)
      predictions = pd.read_csv(os.path.join(args.pred_dir, process_file))
      
      # mapping patient labels to output score
      info_pred = predictions.copy()
      info_cols = ['F','M','Black', "White", "Yes", 'No']
      cols = [c for c in info_pred.columns if c not in ['patient_id']]
      info_pred = info_pred.rename(columns={c:f"{c} score" for c in cols})
      for c in info_cols:
          info_pred[c] = info_pred['patient_id'].map(validation_2_list.set_index("patient_id")[c])
      
      task_gt = info_pred["Yes"]
      task_pred = info_pred["Yes score"]
      info_sub_p = info_pred[info_pred[group_p].isin(['1'])]
      task_gt_p = info_sub_p["Yes"]
      task_pred_p = info_sub_p["Yes score"]
      info_sub_u = info_pred[info_pred[group_u].isin(['1'])]
      task_gt_u = info_sub_u["Yes"]
      task_pred_u = info_sub_u["Yes score"]
      dp = {}     
      dp[f"{group_p}"] = {}
      dp[f"{group_u}"] = {}
      dp["Overall"] = {}
      # privileged group
      tp_1, tn_1, fp_1, fn_1 = get_confusion_matrix(task_pred_p, task_gt_p, threshold=threshold)
      dp[f"{group_p}"][f"TPR_Origin"] = tp_1 / (tp_1+fn_1+1e-8)      
      tp_p, tn_p, fp_p, fn_p = get_confusion_matrix(task_pred_p, task_gt_p, threshold=threshold+optimal_threds) 
      dp[f"{group_p}"][f"TPR_ROC"] = tp_p / (tp_p+fn_p+1e-8)
      # unprivileged group
      tp_2, tn_2, fp_2, fn_2 = get_confusion_matrix(task_pred_u, task_gt_u, threshold=threshold)
      dp[f"{group_u}"][f"TPR_Origin"] = tp_2 / (tp_2+fn_2+1e-8)
      tp_u, tn_u, fp_u, fn_u = get_confusion_matrix(task_pred_u, task_gt_u, threshold=threshold-optimal_threds)
      dp[f"{group_u}"][f"TPR_ROC"] = tp_u / (tp_u+fn_u+1e-8)
      #Overall
      tp, tn, fp, fn = get_confusion_matrix(task_pred, task_gt, threshold=threshold)
      dp["Overall"][f"TPR_Origin"] = tp / (tp+fn+1e-8)
      dp["Overall"][f"TPR_ROC"] = (tp_u+tp_p) / (tp_p+fn_p+tp_u+fn_u+1e-8)

      metrics_summary = pd.DataFrame.from_dict({i: dp[i] 
                           for i in dp.keys()},
                       orient='index')
      metrics_summary.to_csv(os.path.join(sub_dir, f"ROC_mitigation_apply_vali_2.csv"))
      
      
         
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand',type=int)
    parser.add_argument('--sub_dir',type=str)
    parser.add_argument('--pred_dir',type=str)
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--test_subgroup',type=str)
    parser.add_argument('--prediction_file',type=str)
    parser.add_argument('--process_file',type=str)
    parser.add_argument('--info_file',type=str)
    parser.add_argument('--info_file_2',type=str)
    parser.add_argument('--post_processed',default=False,type=bool)
    args = parser.parse_args()
    ROC_mitigation_apply()
    print("Done\n")
