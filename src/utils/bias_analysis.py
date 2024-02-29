import argparse
import os
import pandas as pd
from scipy.special import logit, expit
import torch
import sklearn.metrics as sk_metrics
import numpy as np

import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'monospace'
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titley'] = 1.0 
plt.rcParams['axes.titlepad'] = 8
plt.rcParams["legend.loc"] = 'lower right'
plt.rcParams["legend.handlelength"] = 4.5
plt.rcParams["legend.borderaxespad"] = 0.2
plt.rcParams["legend.fontsize"] = 16

color_dict = {"F":"#dd337c", "M":"#0fb5ae", "Overall":"#57c44f", "Black":"#4046ca", "White":"#f68511"} 
style_dict = {"Default":"-", "M":"--", "F":"-", "Black":"-", "White":"--"} # positive-associated

def get_confusion_matrix(predictions, groundtruth, threshold):
    """ Get counts of true/false positives and true/false negatives. 
    
    Arguments
    =========
    predictions
        numpy array contains prediction scores.
    groundtruth
        numpy array contains ture labels.
    threshold
        Threshold to calculate true/false positives and true/false negatives.

    Returns
    =======
    tp 
        counts of true positive
    tn 
        counts of true negative
    fp 
        counts of false positive
    fn 
        counts of false negative
    
    """
    tp = np.sum(np.logical_and(predictions > threshold, groundtruth == 1))
    tn = np.sum(np.logical_and(predictions <= threshold, groundtruth == 0))
    fp = np.sum(np.logical_and(predictions > threshold, groundtruth == 0))
    fn = np.sum(np.logical_and(predictions <= threshold, groundtruth == 1))
    return tp, tn, fp, fn 

def info_pred_mapping(info:pd.DataFrame, pred:pd.DataFrame)->pd.DataFrame:
    """ Map patient attributes information (e.g. sex, race) to prediction score and labels
    according to the patient id
    
    Arguments
    =========
    info
        Dataframe contains patient attributes info.
    pred
        Dataframe contains model prediction scores.

    Returns
    =======
    pandas.DataFrame
        Dataframe combines patient attributes and predictions.
    
    """
    # drop duplicate patient ids
    info.drop_duplicates(subset="patient_id", keep='first', inplace=True)
    # read prediction result file
    info_pred = pred.copy()
    # mapping patient labels to output score
    info_cols = [c for c in info.columns if c not in ['patient_id', 'Path']]
    for c in info_cols:
        info_pred[c] = info_pred['patient_id'].map(info.set_index("patient_id")[c])
    return info_pred
    

def subgroup_bias_calculation(info_pred:pd.DataFrame, test_list:list, output_file:str, threshold:float)->pd.DataFrame:
    """ Calculate performance and bias measurements for subgroups. 
    
    Arguments
    =========
    info_pred
        Dataframe contains patient attributes and predictions.
    test_list
        List of subgroups for bias measurements calculation.
    output_file
        File path to store calculated performance and bias measurements.
    threshold
        Threshold

    Returns
    =======
    pandas.DataFrame
        Dataframe contains calculated performance and bias measurements.
    
    """
    print("\nStart subgroup bias measurements")
    # nuanced auroc and AEG
    subgroup_df = info_pred[[test_list[0],test_list[1]]].copy()
    true_label = info_pred[['label']].copy()
    pred_prob = info_pred[['score']].copy()
    # fairness measurements
    dp = {}
    for grp in test_list:
        info_sub = info_pred.loc[info_pred[grp]==1]
        task_gt = info_sub['label']
        task_pred = info_sub['score']
        dp[f"{grp}"] = {}
        tp, tn, fp, fn = get_confusion_matrix(task_pred, task_gt, threshold)
        # Demographic Parity criteria
        dp[f"{grp}"]['Demographic Parity (thres)'] = (tp+fp) / (tp+tn+fp+fn+1e-8)
        # Equalized Odds criteria (sensitivity)
        dp[f"{grp}"]['TPR'] = tp / (tp+fn+1e-8)
        # AUROC
        dp[f"{grp}"]['AUROC'] = sk_metrics.roc_auc_score(y_score=task_pred, y_true=task_gt)
        # Overall AUROC for COVID
        dp[f"{grp}"]['Overall AUROC'] = sk_metrics.roc_auc_score(y_score=info_pred["score"], y_true=info_pred["label"])
        # AUROC for subgroup classification
        dp[f"{grp}"]['AUROC_subgroup'] = sk_metrics.roc_auc_score(y_score=info_pred["score"], y_true=info_pred[grp])
            
    return dp
  
def analysis(args):
    exp_dir = args.exp_dir  
    test_info = pd.read_csv(args.testing_info_file)    
    group_dict = {'sex':['M','F'], 'race':['White','Black']}
    test_list = group_dict.get(args.test_subgroup)
    prev_diff_p = []
    prev_diff_n = []
    results_p = []
    results_n = []
    # # calculate bias measurements
    for exp in exp_list:
        prev = int(''.join(filter(str.isdigit, exp)))
        prev_diff = 2 * prev - 1 
        test_result = pd.read_csv(os.path.join(main_dir, exp, args.testing_result_file), sep='\t')
        test_info_result = info_pred_mapping(test_info, test_result)
        output_file = os.path.join(exp_dir, 'subgroup_bias_measure.csv')
        metric_summary = subgroup_bias_calculation(test_info_result, test_list, output_file, args.threshold)
        if prev_diff > 0:
            prev_diff_p.append(prev_diff)
            results_p.append(metric_summary)
        elif prev_diff < 0:
            prev_diff_n.append(-prev_diff)
            results_n.append(metric_summary)
        else:
            prev_diff_p.append(prev_diff)
            prev_diff_n.append(prev_diff)
            results_p.append(metric_summary)
            results_n.append(metric_summary)
    results_p = [x for _,x in sorted(zip(prev_diff_p,results_p))]
    prev_diff_p = sorted(prev_diff_p)
    results_n = [x for _,x in sorted(zip(prev_diff_n,results_n))]
    prev_diff_n = sorted(prev_diff_n)
    dp_p_0 = [x for x in results_p[test_list[0]]['Demographic Parity (thres)']]
    dp_n_0 = [x for x in results_n[test_list[0]]['Demographic Parity (thres)']]
    dp_p_1 = [x for x in results_p[test_list[1]]['Demographic Parity (thres)']]
    dp_n_1 = [x for x in results_n[test_list[1]]['Demographic Parity (thres)']]
    auc_p_0 = [x for x in results_p[test_list[0]]['AUROC']]
    auc_n_0 = [x for x in results_n[test_list[0]]['AUROC']]
    auc_p_1 = [x for x in results_p[test_list[1]]['AUROC']]
    auc_n_1 = [x for x in results_n[test_list[1]]['AUROC']]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(prev_diff_p, dp_p_0, c=color_dict[test_list[0]])
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--main_dir',type=str, required=True)
    parser.add_argument('-e', '--exp_list', nargs='+', default=[], required=True)
    parser.add_argument('-r', '--testing_result_file',type=str, required=True)
    parser.add_argument('-i', '--testing_info_file',type=str, required=True)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-s', '--test_subgroup', type=str, required=True)
    args = parser.parse_args()
    analysis(args)
    
    print("\nDone\n")

