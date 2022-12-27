"""Entry-point script to train models."""
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
# from decision_boundaries import get_planeloader, plot_decision_boundaries
import os
import pandas as pd
import sklearn.metrics as sk_metrics
import numpy as np
import json


analysis_subgroups = ['M','F','Black_or_African_American', 'White', "Yes", "No"]

def get_last_iter(model_dir):
    model_iters = {}
    for fp in os.listdir(model_dir):    
        if not fp.endswith(".pth.tar"):
            continue
        elif 'best' in fp:
            continue
        iter_num = int(fp.replace("iter_","").replace(".pth.tar", ""))
        model_iters[iter_num] = fp
    # find the largest iter number
    x = max(model_iters.keys())
    iter_fp = os.path.join(model_dir,model_iters[x])
    return iter_fp

def test(args, subgroup_analysis=True):
    """Run model testing."""
    # # ADJUSTMENT - getting the last_iter
    # args.model_args.ckpt_path = get_last_iter(Path(args.model_args.ckpt_path).parent)
    # TODO: invesitage logger_args.results_dir (set to results/test?)
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
    bulk_args = args.bulk_args
    by_patient = args.by_patient
    
    # get datasets
    if bulk_args.datasets != 'validation':
        eval_datasets= bulk_args.datasets.split(",")
        print(eval_datasets)
    eval_files = {}
    # get step #
    logger_dir_list = str(logger_args.save_dir).split("/")[-1].split("_")
    step_num = int(logger_dir_list[logger_dir_list.index("step") + 1])
    # get total steps from tracking information
    tracking_file = os.path.join("/".join(str(logger_args.save_dir).split("/")[:-1]), "tracking.log")
    with open(tracking_file, 'rb') as fp:
        tracking_info = json.load(fp)
    total_steps = int(tracking_info['Partition']['steps'])
    # get validation sets for every step
    validation_files = {}
    val_dir = "/".join(str(logger_args.save_dir).split("/")[:-1])
    for n in range(total_steps):
        validation_files[n] = os.path.join(val_dir, f"step_{n}_validation.csv")
    # get the joint validation file -> independent test
    eval_files['independent_test'] = os.path.join(val_dir, 'joint_validation.csv')
    # get independent test sets from other repositories
    if bulk_args.datasets != 'validation':
        for eval_ds in eval_datasets:
            eval_files[eval_ds] = os.path.join(bulk_args.eval_folder, f"{eval_ds}.csv")

    # NEW custom metrics ========
    # # 1.1 -> just current validation
    eval_files['validation'] = validation_files[step_num]
    # # 1.2 (prep) -> will need to know this model's evaluation on next step's validation
    if step_num < (total_steps-1):
        eval_files['next_validation'] = validation_files[step_num+1]
    # # 2.1 -> all previous validation + current validation
    if step_num > 0:
        # check if this exists already
        custom_fp = os.path.join(val_dir, f'custom_2_1__step_{step_num}.csv')
        eval_files['custom_2_1'] = custom_fp
        if not os.path.exists(custom_fp):
            temp_dfs = []
            for i in range(step_num):
                temp_dfs.append(pd.read_csv(validation_files[i]))
            temp_df = pd.concat(temp_dfs)
            temp_df.to_csv(custom_fp, index=None)
    # # 3.1 -> all previous validation + current validation + next validation
    if step_num > 0 and step_num <(total_steps-1):
        # check if this exists already
        custom_fp = os.path.join(val_dir, f'custom_3_1__step_{step_num}.csv')
        eval_files['custom_3_1'] = custom_fp
        if not os.path.exists(custom_fp):
            temp_dfs = []
            for i in range(step_num+1):
                temp_dfs.append(pd.read_csv(validation_files[i]))
            temp_df = pd.concat(temp_dfs)
            temp_df.to_csv(custom_fp, index=None)
    to_rm = []
    for eval_ds in eval_files:
        if not os.path.exists(eval_files[eval_ds]):
            print(f"Could not find an existing summary file for {eval_ds}")
            to_rm.append(eval_ds)
            # return
    for e in to_rm:
        eval_files.pop(e)
    # Get logger.
    logger = Logger(logger_args.log_path,
                    logger_args.save_dir,
                    logger_args.results_dir)

    # Get image paths corresponding to predictions for logging
    
    paths = None

    if model_args.config_path is not None:
        # Instantiate the EnsemblePredictor class for obtaining
        # model predictions.
        predictor = EnsemblePredictor(config_path=model_args.config_path,
                                      model_args=model_args,
                                      data_args=data_args,
                                      gpu_ids=args.gpu_ids,
                                      device=args.device,
                                      logger=logger)
        # Obtain ensemble predictions.
        # Caches both individual and ensemble predictions.
        # We always turn off caching to ensure that we write the Path column.
        predictions, groundtruth, paths = predictor.predict(cache=False,
                                                            return_paths=True,
                                                            all_gt_tasks=True)
    else:
        # Load the model at ckpt_path.
        ckpt_path = model_args.ckpt_path
        ckpt_save_dir = Path(ckpt_path).parent
        model_uncertainty = model_args.model_uncertainty
        # Get model args from checkpoint and add them to
        # command-line specified model args.
        model_args, transform_args\
            = ModelSaver.get_args(cl_model_args=model_args,
                                  dataset=data_args.dataset,
                                  ckpt_save_dir=ckpt_save_dir,
                                  model_uncertainty=model_uncertainty)
        
        # TODO JBY: in test moco should never be true.
        model_args.moco = args.model_args.moco
        model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                                 gpu_ids=args.gpu_ids,
                                                 model_args=model_args,
                                                 is_training=False)
        print(ckpt_info)
        # Instantiate the Predictor class for obtaining model predictions.
        predictor = Predictor(model=model, device=args.device, code_dir=args.code_dir)
        # Get phase loader object.
        return_info_dict = False
        if by_patient:
            return_info_dict = True
        
        
        # # =========================================================
        # switching tasks
        # data_args.metric_name = 'custom-AUROC'
        # data_args.custom_tasks = 'custom-tasks'
        # set up summary
        All_AUROCs = {}
        # loop through datasets
        for ds, ds_file in eval_files.items():
            if by_patient:
                pred_fp = os.path.join(logger_args.save_dir, "results",f"{ds}_by_patient_predictions.csv")
                gt_fp = os.path.join(logger_args.save_dir, "results", f"{ds}_by_patient_groundtruth.csv")
            else:
                # TEMP
                pred_fp = os.path.join(logger_args.save_dir, "results",f"{ds}_predictions.csv")
                gt_fp = os.path.join(logger_args.save_dir, "results", f"{ds}_groundtruth.csv")
                # pred_fp = os.path.join(logger_args.save_dir, "results",f"{ds}_predictions_last_iter.csv")
                # gt_fp = os.path.join(logger_args.save_dir, "results", f"{ds}_groundtruth_last_iter.csv")
            if os.path.exists(pred_fp):
                print(f"{ds} evaluation read from file")
                predictions = pd.read_csv(pred_fp)
                groundtruth = pd.read_csv(gt_fp)
            else:
                print(f"===== Evaluating {ds} =====")
                data_args.test_csv = ds_file
                loader = get_loader(phase=data_args.phase,
                                    data_args=data_args,
                                    transform_args=transform_args,
                                    is_training=False,
                                    return_info_dict=return_info_dict,
                                    logger=logger)
            
                # Obtain model predictions.
                if return_info_dict:
                    predictions, groundtruth, paths = predictor.predict(loader, by_patient=by_patient)
                else:
                    predictions, groundtruth = predictor.predict(loader, by_patient=by_patient)
                # save predictions and ground truth ========= 
                predictions.to_csv(pred_fp)
                groundtruth.to_csv(gt_fp)
            if subgroup_analysis:
                patient_info_fp = os.path.join(logger_args.save_dir, 'results', f"{ds}_patient_information.csv")
                if os.path.exists(patient_info_fp):
                    patient_info = pd.read_csv(patient_info_fp)
                else:
                    # determine the subgroups fo each patient in the ds file
                    df = pd.read_csv(ds_file)
                    patient_info = df.drop(['Path'], axis=1)
                    patient_info = patient_info.groupby(['patient_id']).mean()
                    # drop patients that aren't in the subgroups that we are looking at
                    # TODO: non-binary class functionality
                    patient_info['avg'] = patient_info[analysis_subgroups].mean(axis=1)
                    patient_info = patient_info[patient_info['avg'] == 0.5]
                    print(patient_info[analysis_subgroups+['avg']].head(5))
                    drop_cols = [col for col in patient_info.columns if col not in analysis_subgroups]
                    patient_info = patient_info.drop(drop_cols, axis=1)
                    patient_info.to_csv(patient_info_fp)
            # if groundtruth has more columns than prediction -> adjust
            groundtruth = groundtruth[groundtruth.columns.intersection(predictions.columns)]

            # get AUROCs
            AUROC_dict = {}
            # print(f"TASKS: {model_args.__dict__[TASKS]}")
            # for task in model_args.__dict__[TASKS]:
            #     groundtruth[task] = groundtruth[task].astype(int)
            #     # don't use rows with missing values (-1)
            #     task_gt = groundtruth[groundtruth[task] >= 0]
            #     task_pred = predictions[groundtruth[task] >= 0]
            #     if len(task_gt) <= 1:
            #         continue
            #     # get overall AUROC
            #     if sum(task_gt[task]) == 0: # None in gt
            #         AUROC_dict[f'{task} (overall)'] = nan
            #     elif sum(task_gt[task]) == len(groundtruth): # all gt
            #         AUROC_dict[f'{task} (overall)'] = inf
            #     else:
            #         AUROC_dict[f'{task} (overall)'] = sk_metrics.roc_auc_score(y_true=task_gt[task], y_score=task_pred[task])
            #     if subgroup_analysis:
            #         for sub in analysis_subgroups:
            #             subgroup = patient_info[patient_info[sub] == 1]
            #             if len(subgroup) == 0:
            #                 continue
            #             # return
            #             if 'patient_id' not in subgroup.columns:
            #                 if subgroup.index.name == 'patient_id':
            #                     sub_pids = subgroup.index.to_list()
            #                 else:
            #                     print("patient_id not found in columns or index")
            #                     print(subgroup.columns)
            #                     print(subgroup.head(5))
            #                     return
            #             else:
            #                 sub_pids = subgroup['patient_id'].to_list()
            #             # get AUROC for patients of this subgroup
            #             print(groundtruth.head(5))
            #             sub_gt = groundtruth[groundtruth['patient_id'].isin(sub_pids)]
            #             sub_pred = predictions[predictions['patient_id'].isin(sub_pids)]
            #             if sum(sub_gt[task]) == 0: # None in gt
            #                 AUROC_dict[f'{task} (within {sub})'] = nan
            #             elif sum(sub_gt[task]) == len(sub_gt): # all gt
            #                 AUROC_dict[f"{task} (within {sub})"] = inf
            #             else:
            #                 AUROC_dict[f"{task} (within {sub})"] = sk_metrics.roc_auc_score(y_true=sub_gt[task], y_score=sub_pred[task])
                

                # for key, vals in CUSTOM_TASK_SUBSETS.items():
                #     if task in vals:
                #         continue
                #     for val in vals:
                #         #print(f"{task}/{val}")
                #         # get only samples where gt val=1
                #         val_idxs = task_gt.index[task_gt[val] == 1].tolist()
                #         val_gt = task_gt.loc[val_idxs]
                #         val_pred = task_pred.loc[val_idxs]
                #         if sum(val_gt[task]) == 0:
                #             # AUROC_dict[f"{task} (within {val})"] = "none in gt"
                #             AUROC_dict[f"{task} (within {val})"] = np.NaN
                #         elif sum(val_gt[task]) == len(val_gt[task]):
                #             # AUROC_dict[f"{task} (within {val})"] = "all gt"
                #             AUROC_dict[f"{task} (within {val})"] = np.NaN
                #         else:   
                #             AUROC_dict[f"{task} (within {val})"] = sk_metrics.roc_auc_score(y_true=val_gt[task], y_score=val_pred[task])
            
            # print("============")
            # print(ds)
            # print(AUROC_dict)
            # print("============")
            # update overall summary
            All_AUROCs[ds] = AUROC_dict
        # create summary df
        overall_summ = pd.DataFrame(columns=AUROC_dict.keys(), dtype=float)
        for ds, dic in All_AUROCs.items():
            overall_summ.loc[ds] = dic
        print(overall_summ)
        overall_summ.to_csv(os.path.join(logger_args.save_dir, "AUROC_summary.csv"))
        
        
    #     # save predictions and ground truth ========= # TODO
    #     r = 1
    #     repo = "validation"
    #     predictions.to_csv(f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/RAND_{r}_{repo}_predictions.csv")
    #     groundtruth.to_csv(f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/RAND_{r}_{repo}_groundtruth.csv")
    #     # ================== Decision Boundaries ==================
    #     print('=======================')
    #     print(predictions)
    #     #plot_decision_boundaries(predictions, loader, loader.dataset.base_labels)
        
    #     # print(predictions[CHEXPERT_COMPETITION_TASKS])
    #     if model_args.calibrate:
    #         #open the json file which has the saved parameters
    #         import json
    #         with open(CALIBRATION_FILE) as f:
    #             data = json.load(f)
    #         i = 0
    #         #print(predictions)
    #         import math
    #         def sigmoid(x):
    #             return 1 / (1 + math.exp(-x))

    #         for column in predictions:
    #             predictions[column] = predictions[column].apply \
    #                                   (lambda x: sigmoid(x * data[i][0][0][0] \
    #                                   + data[i][1][0]))
    #             i += 1
    #         # print(predictions[CHEXPERT_COMPETITION_TASKS])
    #         #run forward on all the predictions in each row of predictions

    # # Log predictions and groundtruth to file in CSV format.
    
    # logger.log_predictions_groundtruth(predictions, groundtruth, paths)
    
    # if not args.inference_only:
    #     print('evaluating...')
    #     # Instantiate the evaluator class for evaluating models.
    #     evaluator = Evaluator(logger)
    #     # Get model metrics and curves on the phase dataset.
    #     metrics, curves = evaluator.evaluate_tasks(groundtruth, predictions)
    #     # Log metrics to stdout and file.
    #     logger.log_stdout(f"Writing metrics to {logger.metrics_path}.")
    #     # change save location of results
    #     logger.metrics_csv_path = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/RAND_14_COVID_19_NY_SBU_summary.csv"
    #     logger.log_metrics(metrics, save_csv=True)

    # # TODO: make this work with ensemble
    # # TODO: investigate if the eval_loader can just be the normal loader here
    # if logger_args.save_cams:
    #     cams_dir = logger_args.save_dir / 'cams'
    #     print(f'Save cams to {cams_dir}')
    #     save_grad_cams(args, loader, model,
    #                    cams_dir,
    #                    only_competition=logger_args.only_competition_cams,
    #                    only_top_task=False)

    logger.log("=== Testing Complete ===")
    # # Produce other visuals
    # # TODO: This causes "unexpected error to scripts"
    # # raise NotImplementedError()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    print("Start test...")
    test(parser.parse_args())
