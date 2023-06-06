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

custom_subgroups ={ 
    'sex':{'M','F'},
    'race':{'White', 'Black'},
    'COVID_positive':{'Yes', 'No'},
    'modality':{'CR', 'DX'}
}

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

def test(args):
    """Run model testing."""
    # TODO: invesitage logger_args.results_dir (set to results/test?)
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
    bulk_args = args.bulk_args
    by_patient = args.by_patient
    # ADJUSTMENT - getting the last_iter
    model_args.ckpt_path = get_last_iter(Path(args.model_args.ckpt_path).parent)
    
    # get datasets
    if bulk_args.datasets != 'validation':
        eval_datasets= bulk_args.datasets.split(",")
        print(eval_datasets)
    eval_files = {}
    # get step #
    # logger_dir_list = str(logger_args.save_dir).split("/")[-1].split("_")
    # step_num = int(logger_dir_list[logger_dir_list.index("step") + 1])
    # get total steps from tracking information
    # tracking_file = os.path.join("/".join(str(logger_args.save_dir).split("/")[:-1]), "tracking.log")
    # with open(tracking_file, 'rb') as fp:
    #    tracking_info = json.load(fp)
    # total_steps = int(tracking_info['Partition']['steps'])
    # get validation sets for every step
    validation_files = []
    val_dir = bulk_args.eval_folder
    #validation_files.append(os.path.join(val_dir, f"test.csv"))
    validation_files.append(os.path.join(val_dir, f"validation_1_image.csv"))
    validation_all = pd.read_csv(validation_files[0])
    eval_files['all'] = validation_files[0]
    #subgroup_for_eval = ['race']
    #for sub in subgroup_for_eval:
    #    for grp in custom_subgroups[sub]:
    #        validation_sub = validation_all[validation_all[grp].isin(['1'])]
    #        validation_sub.to_csv(os.path.join(val_dir, f"validation_2_{grp}.csv"))
    #        validation_files.append(os.path.join(val_dir, f"validation_2_{grp}.csv"))
    #        eval_files[grp] = validation_files[-1]

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
        return_info_dict = True
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
                pred_fp = os.path.join(logger_args.save_dir, "results",f"by_patient_predictions.csv")
                gt_fp = os.path.join(logger_args.save_dir, "results", f"by_patient_groundtruth.csv")
            else:
                # TEMP
                # pred_fp = os.path.join(logger_args.save_dir, "results",f"{ds}_predictions.csv")
                # gt_fp = os.path.join(logger_args.save_dir, "results", f"{ds}_groundtruth.csv")
                pred_fp = os.path.join(logger_args.save_dir, "results",f"predictions_path.csv")
                gt_fp = os.path.join(logger_args.save_dir, "results", f"groundtruth_path.csv")
            if os.path.exists(pred_fp):
                print(f"{ds} evaluation read from file")
                predictions = pd.read_csv(pred_fp)
                groundtruth = pd.read_csv(gt_fp)
            else:
                print(f"===== Evaluating {ds} =====")
                # if 'custom' in ds:
                #     # need to create custom csvs to evaluate
                #     custom_save_name = os.path.join("/".join(str(logger_args.save_dir).split("/")[:-1]), f"{ds}_step_{step_num}.csv")
                #     if not os.path.exists(custom_save_name):
                #         custom_files = []
                #         for file in ds_file:
                #             custom_files.append(pd.read_csv(file, index_col=0))
                #         combined_df = pd.concat(custom_files)
                #         combined_df.reset_index()
                #         combined_df.to_csv(custom_save_name)
                #     data_args.test_csv = custom_save_name
                # else:
                #     data_args.test_csv = ds_file
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
            
            # if groundtruth had=s more columns than prediction -> adjust
            groundtruth = groundtruth[groundtruth.columns.intersection(predictions.columns)]

            # get AUROCs
            AUROC_dict = {}
            # print(f"TASKS: {model_args.__dict__[TASKS]}")
            for task in model_args.__dict__[TASKS]:
                print(task)
                # don't use rows with missing values (-1)
                task_gt = groundtruth[groundtruth[task] >= 0]
                task_pred = predictions[groundtruth[task] >= 0]
                # print(task_gt[task].head(5))
                # print(task_pred[task].head(5))
                if len(task_gt) <= 1:
                    continue
                # get overall AUROC
                if sum(task_gt[task]) == 0:
                    # AUROC_dict[f'{task} (overall)'] = "none in gt"
                    AUROC_dict[f'{task} (overall)'] = nan
                elif sum(task_gt[task]) == len(groundtruth):
                    # AUROC_dict[f'{task} (overall)'] = "all gt"
                    AUROC_dict[f'{task} (overall)'] = inf
                else:
                    AUROC_dict[f'{task} (overall)'] = sk_metrics.roc_auc_score(y_true=task_gt[task], y_score=task_pred[task])
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
        overall_summ.to_csv(os.path.join(logger_args.save_dir, "AUROC_summary_subgroup.csv"))
        
        
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
