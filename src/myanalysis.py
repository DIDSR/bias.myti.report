import argparse
import os
import pandas as pd
from scipy.special import logit
from math import inf, nan
from nuancedmetric import *
import torch
import sklearn.metrics as sk_metrics
import numpy as np

def get_confusion_matrix(predictions, groundtruth, threshold):
    tp = np.sum(np.logical_and(predictions > threshold, groundtruth == 1))
    tn = np.sum(np.logical_and(predictions <= threshold, groundtruth == 0))
    fp = np.sum(np.logical_and(predictions > threshold, groundtruth == 0))
    fn = np.sum(np.logical_and(predictions <= threshold, groundtruth == 1))
    return tp, tn, fp, fn 

def info_pred_mapping(info, pred):
    '''
    map patient subgroup information (e.g. sex, race) to prediction score and labels
    according to the patient id 
    '''
    # drop duplicate patient ids
    info.drop_duplicates(subset="patient_id", keep='first', inplace=True)
    # read prediction result file
    info_pred = pred.copy()
    # mapping patient labels to output score
    info_cols = [c for c in info.columns if c not in ['patient_id', 'Path']]
    for c in info_cols:
        info_pred[c] = info_pred['patient_id'].map(info.set_index("patient_id")[c])
    return info_pred
    
def model_ensemble(main_dir, exp_name, prediction_file, model_number=10):
    '''
    load predicted scores from each model, and average them together into one
    ensembled prediction score. Choose to ensemble rather than logits because
    it is slightly favorable suggested by the paper.
    The output score is saved under the randome_state_0 directory.
    '''
    predictions_all = []
    # load scores from models
    for RD in range(model_number):
        pred = pd.read_csv(os.path.join(main_dir, f'{exp_name}_RD_{RD}', prediction_file), sep='\t')
        predictions = pred[['score']].copy()
        predictions = predictions.rename(columns={'score':f'score_{RD}'}) 
        predictions_all.append(predictions)
    
    #compute average scores
    ensemble_scores = pd.concat(predictions_all, axis=1, ignore_index=False)
    ensemble_scores = ensemble_scores.loc[:,~ensemble_scores.columns.duplicated()]
    cols = [f'score_{RD}' for RD in range(model_number)]
    ensemble_scores['score'] = ensemble_scores.loc[:, cols].mean(axis = 1)    
    logits_new = logit(ensemble_scores['score'])
    #output the average score to a new file
    pred['score'] = ensemble_scores['score']
    pred['logits'] = logits_new
    pred.to_csv(os.path.join(main_dir, f'{exp_name}_RD_0', f'ensemble_{prediction_file}'),index=False)
    return pred
    
def ROC_mitigation(validation_info_pred, test_list, threshold=0.5, output_file=None):
    '''
    reject ojective classification method for bias mitigation.Uuse the validation dataset to find 
    prviliged and unprivileged subgroup, as well as the best threshold.
    searching results can optionally save as a csv file.
    '''          
    # determine the privileged group
    print("Beginning bias mitigation using reject ojective classification")
    dp = {}     
    for grp in test_list:
        info_sub = validation_info_pred.loc[(validation_info_pred[grp]==1)]
        tp, tn, fp, fn = get_confusion_matrix(info_sub['score'], info_sub['label'], threshold)
        dp[f"{grp}"] = {}
        dp[f"{grp}"]['TPR'] = tp / (tp+fn)          
    if dp[test_list[0]]['TPR'] > dp[test_list[1]]['TPR']:
        group_p = test_list[0]
        group_u = test_list[1]
    else:
        group_p = test_list[1]
        group_u = test_list[0]
    # searching the optimal thresholds
    info_sub_p = validation_info_pred.loc[(validation_info_pred[group_p]==1)]
    info_sub_u = validation_info_pred.loc[(validation_info_pred[group_u]==1)]
    optimal_threds = 0
    min_diff = 1    
    dp["Overall"] = {}    
    for threds in np.linspace(0, 0.49, 50):
        # privileged group
        tp_p, tn_p, fp_p, fn_p = get_confusion_matrix(info_sub_p['score'], info_sub_p['label'], threshold=threshold+threds)
        sen_1 = tp_p / (tp_p+fn_p+1e-8)
        dp[f"{group_p}"][f"TPR_t_{threds}"] = sen_1
        # unprivileged group
        tp_u, tn_u, fp_u, fn_u = get_confusion_matrix(info_sub_u['score'], info_sub_u['label'], threshold=threshold-threds)
        sen_2 = tp_u / (tp_u+fn_u+1e-8)
        dp[f"{group_u}"][f"TPR_t_{threds}"] = sen_2
        #Overall
        dp["Overall"][f"TPR_t_{threds}"] = (tp_u+tp_p) / (tp_p+fn_p+tp_u+fn_u+1e-8)
        if abs(sen_1-sen_2) < min_diff:
            min_diff = abs(sen_1-sen_2)
            optimal_threds = threds                
    print("The optimal threshold is")
    print(optimal_threds)        
    metrics_summary = pd.DataFrame.from_dict({i: dp[i] 
                           for i in dp.keys()},
                       orient='index')
    if output_file:
        metrics_summary.to_csv(output_file)
    return [threshold + optimal_threds, threshold - optimal_threds], [group_p, group_u]

def subgroup_bias_calculation(info_pred, test_list, output_file, thresholds):
    '''
    function to calculate measurements for subgroup bias
    input prediction_info file, subgroups to test and threshold
    save the calculated measurements to a csv file
    '''
    # nuanced auroc and AEG
    subgroup_df = info_pred[[test_list[0],test_list[1]]].copy()
    true_label = info_pred[['label']].copy()
    pred_prob = info_pred[['score']].copy()
    nuance = NuancedROC()
    nuance_result = nuance.score(true_label, pred_prob, subgroup_df)
    aeg = AEG()
    aeg_result = aeg.score(true_label, pred_prob, subgroup_df)

    # fairness measurements
    dp = {}
    for grp, threds in zip(test_list, thresholds):
        info_sub = info_pred.loc[info_pred[grp]==1]
        task_gt = info_sub['label']
        task_pred = info_sub['score']
        dp[f"{grp}"] = {}
        tp, tn, fp, fn = get_confusion_matrix(task_pred, task_gt, threds)
        # Average Score
        dp[f"{grp}"]['Average Score'] = np.mean(task_pred)
        # Demographic Parity criteria
        dp[f"{grp}"]['Demographic Parity (thres)'] = (tp+fp) / (tp+tn+fp+fn+1e-8)
        # Equalized Odds criteria (sensitivity)
        dp[f"{grp}"]['TPR'] = tp / (tp+fn+1e-8)
        # Predictive Rate Parity
        dp[f"{grp}"]['PPV'] = tp / (tp+fp+1e-8)
        # specificity
        dp[f"{grp}"]['TNR'] = tn / (tn+fp+1e-8)
        # AUROC
        dp[f"{grp}"]['AUROC'] = sk_metrics.roc_auc_score(y_score=task_pred, y_true=task_gt)
        # Overall AUROC for COVID
        dp[f"{grp}"]['Overall AUROC'] = sk_metrics.roc_auc_score(y_score=info_pred["score"], y_true=info_pred["label"])
        # AUROC for subgroup classification
        dp[f"{grp}"]['AUROC_subgroup'] = sk_metrics.roc_auc_score(y_score=info_pred["score"], y_true=info_pred[grp])
        # NLL (uncertainty estimation)
        p = torch.tensor(info_pred['score'])
        l = torch.tensor(info_pred['label'])
        nll_criterion = torch.nn.BCELoss()
        dp[f"{grp}"]['NLL_overall'] = nll_criterion(p, l.double()).item()
        p_sub = torch.tensor(task_pred.reset_index(drop=True))
        l_sub = torch.tensor(task_gt.reset_index(drop=True))
        dp[f"{grp}"]['NLL'] = nll_criterion(p_sub, l_sub.double()).item()
        # Save the thresholds
        dp[f"{grp}"]['Thresholds'] = threds      
            
    fairness_summary = pd.DataFrame.from_dict({i: dp[i] 
                           for i in dp.keys()},
                       orient='index')
    metrics_summary = pd.concat([fairness_summary, nuance_result, aeg_result], join='outer',axis=1)
    metrics_summary.to_csv(output_file)
    return metrics_summary
    
def analysis(args):
    main_dir = args.main_dir
    exp_name = args.exp_name
    test_info = pd.read_csv(args.testing_info_file)
    vali_info = pd.read_csv(args.validation_info_file)
    test_list = args.test_subgroup
    threshold = args.threshold
    
    # # ensemble prediction results
    ensembled_test = model_ensemble(main_dir, exp_name, args.testing_file, args.model_number)
    
    # # calculate original bias measurements
    test_info_pred = info_pred_mapping(test_info, ensembled_test)
    output_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'subgroup_bias_measure.csv')
    metrics_summary = subgroup_bias_calculation(test_info_pred, test_list, output_file, [threshold, threshold])
    
    # # apply post processing bias mitigation methods if the user choose to    
    if args.post_bias_mitigation is not None:
        ensembled_vali = model_ensemble(main_dir, exp_name, args.validation_file, args.model_number)
        validation_info_pred = info_pred_mapping(vali_info, ensembled_vali)
        if args.post_bias_mitigation == 'reject_object_class':
            roc_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'validation_roc_searching.csv')
            optim_threds, subgroups = ROC_mitigation(validation_info_pred, test_list, threshold, roc_file)
            roc_output_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'subgroup_bias_measure_post_roc.csv')
            metrics_summary = subgroup_bias_calculation(test_info_pred, subgroups, roc_output_file, optim_threds)
        else:
            raise RuntimeError('Current post processing bias mitigation method is not supported!')
        
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir',type=str)
    parser.add_argument('--exp_name',type=str)
    parser.add_argument('--model_number',type=int, default=10)
    parser.add_argument('--validation_file',type=str)
    parser.add_argument('--testing_file',type=str)
    parser.add_argument('--validation_info_file',type=str)
    parser.add_argument('--testing_info_file',type=str)
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('--test_subgroup',nargs='+',type=str)
    parser.add_argument('--post_bias_mitigation', help="which post processing bias mitigation method to use: 'reject_object_class'")
    args = parser.parse_args()
    analysis(args)
    
    print("Done\n")

