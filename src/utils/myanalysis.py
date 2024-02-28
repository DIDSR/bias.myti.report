import argparse
import os
import pandas as pd
from scipy.special import logit, expit
from utils import *
import torch
import sklearn.metrics as sk_metrics
import numpy as np

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
    
def model_ensemble(main_dir:str, exp_name:str, prediction_file:str, model_number:int=10)->pd.DataFrame:
    """ Gets the ensembled prediction scores from models with different initial random seeds.
    
    Arguments
    =========
    main_dir
        Main path of the experiment.
    exp_name
        The name indicating the current experiment.
    prediction_file
        Name of the file which contains prediction scores.
    model_number
        Number of models to ensemble.

    Returns
    =======
    pandas.DataFrame
        Dataframe contains ensembled prediction scores and logits.
    
    """
    
    print("\nStart model ensemble")
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
    print("\nModel ensemble Done\n")
    return pred
    

def calibrate_model(ensembled_vali:pd.DataFrame, ensembled_test:pd.DataFrame, output_file:str=None):
    """ Calibrate models using temperature scaling.
    
    Arguments
    =========
    ensembled_vali
        Dataframe contains predictions of validation set.
    ensembled_test
        Dataframe contains predictions of testing set.
    output_file
        File path to store calibration metrics.

    Returns
    =======
    calib_vali
        Dataframe contains validation prediction scores after calibration.
    calib_test
        Dataframe contains testing prediction scores after calibration.
    
    """
    print("\nStart model calibration")
    # # get uncalibrated logits and labels
    vali_logit = ensembled_vali['logits'].values
    vali_label = ensembled_vali['label'].values
    test_logit = ensembled_test['logits'].values
    test_label = ensembled_test['label'].values
    # # temperature scaling, compute temperature from validation set    
    model_calib = CalibratedModel()
    vali_results = [vali_logit, vali_label]
    t = model_calib.set_temperature(vali_results)
    calib_vali_logit = vali_logit / t
    calib_vali_score = expit(calib_vali_logit)
        
    # # apply the temperature to test set
    #metrics before calibration
    metric_pre_calib = model_calib.get_metrics(test_logit, test_label)
    metric_pre_calib.insert(0, 'Status', 'Before Calibration')
    calib_test_logit = test_logit / t
    calib_test_score = expit(calib_test_logit)
    #metrics after calibration
    metric_post_calib = model_calib.get_metrics(calib_test_logit, test_label)
    metric_post_calib.insert(0, 'Status', 'After Calibration')
    if output_file:
        metric_out = pd.concat([metric_pre_calib, metric_post_calib])
        metric_out.to_csv(output_file, index=False)
    # # save results and return
    calib_vali = ensembled_vali.copy()
    calib_vali['logits'] = calib_vali_logit
    calib_vali['score'] = calib_vali_score
    calib_test = ensembled_test.copy()
    calib_test['logits'] = calib_test_logit
    calib_test['score'] = calib_test_score
    print("\nModel calibration Done\n")
    return calib_vali, calib_test
    
    
    
def ROC_mitigation(validation_info_pred:pd.DataFrame, test_list:list, threshold:float=0.5, output_file:str=None):          
    """ Post-processing bias mitigation using reject option classification method.
    
    Arguments
    =========
    validation_info_pred
        Dataframe contains patient attributes and predictions of validation set.
    test_list
        List of subgroups for bias mitigation.
    threshold
        Default threshold
    output_file
        File path to store new threshold searching results.

    Returns
    =======
    threds_list
        lists contains new thresholds for mitigation subgroups.
    group_list
        list specify privilege and unprivileged subgroup
    
    """
    print("\nStart bias mitigation using reject option classification")
    # # determine the privileged group
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
        metrics_summary.to_csv(output_file, index=False)
    print("\nBias mitigation using reject option classification done")
    threds_list = [threshold + optimal_threds, threshold - optimal_threds]
    group_list = [group_p, group_u]
    return threds_list, group_list

    
def calib_eq_odds_mitigation(validation_info_pred:pd.DataFrame, testing_info_pred:pd.DataFrame, test_list:list, output_file:str, rate_list:list)->pd.DataFrame:
    """ Post-processing bias mitigation using calibrated equalized odds method.
    
    Arguments
    =========
    validation_info_pred
        Dataframe contains patient attributes and predictions of validation set.
    testing_info_pred
        Dataframe contains patient attributes and predictions of testing set.
    test_list
        List of subgroups for bias mitigation.
    output_file
        File path to store predictions after implementation of calibrated equalized odds.
    rate_list
        weight lists for FPR and FNR

    Returns
    =======
    pandas.DataFrame
        Dataframe contains prediction scores for testing after implementation of calibrated equalized odds.
    
    """
    print("\nStart bias mitigation using calibrated equalized odds")
    # Create model objects - one for each group, validation and test 
    group_0_vali_data = validation_info_pred[validation_info_pred[test_list[0]] == 1]
    group_1_vali_data = validation_info_pred[validation_info_pred[test_list[0]] == 0]   
    group_0_test_data = testing_info_pred[testing_info_pred[test_list[0]] == 1]
    group_1_test_data = testing_info_pred[testing_info_pred[test_list[0]] == 0]
    
    group_0_vali_model = CalibEqOddsModel(group_0_vali_data['score'].values, group_0_vali_data['label'].values)
    group_1_vali_model = CalibEqOddsModel(group_1_vali_data['score'].values, group_1_vali_data['label'].values)
    group_0_test_model = CalibEqOddsModel(group_0_test_data['score'].values, group_0_test_data['label'].values)
    group_1_test_model = CalibEqOddsModel(group_1_test_data['score'].values, group_1_test_data['label'].values)
    

    # Find mixing rates for equalized odds models
    fp_rate = rate_list[0]
    fn_rate = rate_list[1]
    _, _, mix_rates = CalibEqOddsModel.calib_eq_odds(group_0_vali_model, group_1_vali_model, fp_rate, fn_rate)
    # Apply the mixing rates to the test models
    calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model = CalibEqOddsModel.calib_eq_odds(group_0_test_model,
                                                                                             group_1_test_model,
                                                                                             fp_rate, fn_rate,
                                                                                             mix_rates)
    
    # Reorganize the prediction scores and output as a new file
    pred_1 = pd.DataFrame(list(zip(calib_eq_odds_group_0_test_model.label, calib_eq_odds_group_0_test_model.pred)), columns=['label', 'score'])
    pred_1[test_list[0]] = 1
    pred_1[test_list[1]] = 0    
    pred_2 = pd.DataFrame(list(zip(calib_eq_odds_group_1_test_model.label, calib_eq_odds_group_1_test_model.pred)), columns=['label', 'score'])
    pred_2[test_list[0]] = 0
    pred_2[test_list[1]] = 1    
    pred_all = pd.concat([pred_1, pred_2], axis=0)
    pred_all.to_csv(output_file, index=False)
    print("\nBias mitigation using calibrated equalized odds done")
    return pred_all


def subgroup_bias_calculation(info_pred:pd.DataFrame, test_list:list, output_file:str, thresholds:list)->pd.DataFrame:
    """ Calculate performance and bias measurements for subgroups. 
    
    Arguments
    =========
    info_pred
        Dataframe contains patient attributes and predictions.
    test_list
        List of subgroups for bias measurements calculation.
    output_file
        File path to store calculated performance and bias measurements.
    thresholds
        List of thresholds used for subgroups in test_list.

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
        p = torch.tensor(info_pred['score'].reset_index(drop=True))
        l = torch.tensor(info_pred['label'].reset_index(drop=True))
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
    print("\nSubgroup bias measurements done")
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
    ensembled_vali = model_ensemble(main_dir, exp_name, args.validation_file, args.model_number)

    
    # # run model calibration
    calib_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'calibration_metrics.csv')
    calib_vali, calib_test = calibrate_model(ensembled_vali, ensembled_test, calib_file)
    
    # # calculate original bias measurements
    test_info_pred = info_pred_mapping(test_info, calib_test)
    output_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'subgroup_bias_measure.csv')
    metrics_summary = subgroup_bias_calculation(test_info_pred, test_list, output_file, [threshold, threshold])
    
    # # apply post processing bias mitigation methods if the user choose to    
    if args.post_bias_mitigation is not None:
        validation_info_pred = info_pred_mapping(vali_info, calib_vali)
        if args.post_bias_mitigation == 'reject_option_class':
            # # run reject objective classification bias mitigation methods
            roc_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'validation_roc_searching.csv')
            optim_threds, subgroups = ROC_mitigation(validation_info_pred, test_list, threshold, roc_file)
            roc_output_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'subgroup_bias_measure_post_roc.csv')
            metrics_summary = subgroup_bias_calculation(test_info_pred, subgroups, roc_output_file, optim_threds)
        elif args.post_bias_mitigation == 'calib_eq_odds':
            # # run calibrated equalized odds bias mitigation methods
            calib_eq_odds_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'calib_eq_odds_prediction.csv')
            calib_eq_odds_result = calib_eq_odds_mitigation(validation_info_pred, test_info_pred, test_list, calib_eq_odds_file, args.calib_eq_odds_rates)
            calib_eq_odds_output_file = os.path.join(main_dir, f'{exp_name}_RD_0', 'subgroup_bias_measure_post_calib_eq_odds.csv')
            metrics_summary = subgroup_bias_calculation(calib_eq_odds_result, test_list, calib_eq_odds_output_file, [threshold, threshold])
        else:
            raise RuntimeError('Current post processing bias mitigation method is not supported!')
        
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--main_dir',type=str)
    parser.add_argument('-e', '--exp_name',type=str)
    parser.add_argument('-n', '--model_number',type=int, default=10)
    parser.add_argument('-v', '--validation_file',type=str)
    parser.add_argument('-t', '--testing_file',type=str)
    parser.add_argument('-iv', '--validation_info_file',type=str)
    parser.add_argument('-it', '--testing_info_file',type=str)
    parser.add_argument('--threshold',type=float,default=0.5)
    parser.add_argument('-s', '--test_subgroup',nargs='+',type=str)
    parser.add_argument('-p', '--post_bias_mitigation', 
    help="which post processing bias mitigation method to use: 'reject_option_class', 'calib_eq_odds'")
    parser.add_argument('-r', '--calib_eq_odds_rates',nargs=2,type=float, default=[0.5, 1],
    help="Specify 2 weights for FPR and FNR in calibrated equalized odds mitigation method")
    args = parser.parse_args()
    analysis(args)
    
    print("\nDone\n")

