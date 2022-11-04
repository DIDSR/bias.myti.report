from itertools import product, combinations, combinations_with_replacement
from decision_boundaries import get_planeloader, decision_region_analysis
import pandas as pd
import os
import random
from random import sample
import torch
from args import TestArgParser
from saver import ModelSaver
from predict import Predictor
from constants import *
from data import get_loader
import numpy as np


test_input_classes = {'sex':['M','F'], 'race':['White', 'Black_or_African_American'], "COVID_positive":["Yes", "No"]}
test_output_classes = ['Yes', "No"]
# test_input_classes = {'sex':['M','F'],"COVID_postive":["Yes", "No"]}
abbreviation_table = {
    'Female':"F",
    'Male':"M",
    'CR':"C",
    "DX":"D",
    "White":"W",
    'Black_or_African_American':"B",
    "Yes":"P",# positive
    "No":'N'
}
def get_all_combos(input_classes, consistent_triplet):
    interaction_subgroups = list(product(*input_classes))
    if not consistent_triplet:
        interaction_subgroup_combinations = list(combinations_with_replacement(interaction_subgroups, 3))
    else:
        interaction_subgroup_combinations = [[item]*3 for item in interaction_subgroups]
    return interaction_subgroups, interaction_subgroup_combinations

def get_img_idxs(idxs, allow_image_reuse, n_samples, curr_idxs=[]):
    curr_idxs = [set(l) for l in curr_idxs]
    for n in range(len(curr_idxs), n_samples):
        item = sample(idxs, 3)
        while set(item) in curr_idxs:
            item = sample(idxs, 3)
        curr_idxs.append(set(item))
    return [list(s) for s in curr_idxs]


        

def decision_boundary_setup(groundtruth,
                            input_classes,
                            output_classes,
                            output_dir,
                            num_samples,
                            triplet_classes=None,
                            consistent_triplet=True,
                            allow_image_reuse=True,
                            update_summary_every=5):
    # temp
    # torch.manual_seed(1)
    random.seed(1)
    # TODO: inconsistent triplets
    interaction_subgroups, all_possible_triplet_classes = get_all_combos(input_classes.values(), consistent_triplet=consistent_triplet)
    # set up summary dataframe
    overall_summary_fname = os.path.join(output_dir, 'test_overall_summary.csv') # TODO: naming convention
    if os.path.exists(overall_summary_fname):
        summary_df = pd.read_csv(overall_summary_fname)
    elif consistent_triplet:
        summary_df = pd.DataFrame(columns=['triplet class'] + [key for key in input_classes]+['n_combinations','AUROC'] + [f"%{x}" for x in output_classes])
    else:
        print("WIP summary file")
    if triplet_classes == None:
        triplet_classes = all_possible_triplet_classes
    # get the indexs of the images of each interaction subgroup
    isub_idxs = {}
    for isub in interaction_subgroups:
        temp_df = groundtruth.copy()
        for c in isub:
            temp_df = temp_df[temp_df[c] == 1]
        isub_idxs[isub] = temp_df.index.tolist()
    # get/create overall summary file
    
    for triplet in triplet_classes:
        if len(set(triplet)) > 1: # TODO
            continue
        # check if existing individual summary file
        t_name = "_".join(["".join(i) for i in triplet])
        for i, j in abbreviation_table.items():
            t_name = t_name.replace(i,j)
        indiv_summary_fname = os.path.join(output_dir, f"{t_name}_combinatations.csv")
        if os.path.exists(indiv_summary_fname):
            indiv_df = pd.read_csv(indiv_summary_fname, index_col=0)
            indiv_df  = indiv_df.astype(dtype={'img1 idx':int, 'img2 idx':int, 'img3 idx':int})
            # get idx combinations already in df
            current_idxs = []
            for ii, row in indiv_df.iterrows():
                current_idxs.append([row['img1 idx'], row['img2 idx'], row['img3 idx']])
        else:
            # set up the individual df
            indiv_df = pd.DataFrame(columns=['img1 idx', 'img2 idx', 'img3 idx'] + [f"%{x}" for x in output_classes])
            current_idxs = []
        if len(indiv_df) < num_samples:
            # either new individual df, or num_samples has increased
            img_idxs = get_img_idxs(isub_idxs[isub][:10], allow_image_reuse=allow_image_reuse, n_samples=num_samples, curr_idxs=current_idxs)
            for imgs in img_idxs:
                if len(indiv_df[(indiv_df['img1 idx'] == imgs[0]) & (indiv_df['img2 idx'] == imgs[1]) & (indiv_df['img3 idx'] == imgs[2])] != 0):
                    continue
                indiv_df.loc[len(indiv_df)] = imgs + [None]*len(output_classes)
            indiv_df = indiv_df.astype(dtype={'img1 idx':int, 'img2 idx':int, 'img3 idx':int})
            # print(indiv_df)
            indiv_df.to_csv(indiv_summary_fname)
        # add to summary_df
        if len(summary_df[summary_df['triplet class'] == t_name]) == 0:
            cls = {}
            for x in triplet[0]:
                for key, vals in input_classes.items():
                    if x in vals:
                        cls[key] = x 
            summary_df.loc[len(summary_df)] = [t_name]+[cls[key] for key in input_classes] +[0]*4
        
    print(summary_df)

def trial_setup(data_args, transform_args,  predictor, save_dir, input_classes=test_input_classes, output_classes=test_output_classes, consistent_triplet=True, num_samples=10):
    # determine the possible combinations
    interaction_subgroups, all_possible_triplet_classes = get_all_combos(input_classes.values(), consistent_triplet=consistent_triplet)
    # load groundtruth, limit to only correct classifications
    loader = get_loader(phase = data_args.phase,
                        data_args = data_args,
                        transform_args=transform_args,
                        is_training=False,
                        return_info_dict=True)
    base_predictions, base_groundtruth, base_paths = predictor.predict(loader)
    print(f"Base groundtruth has {len(base_groundtruth)} values")
    for key, vals in input_classes.items():
        cols = [v for v in vals if v in output_classes]
        if len(cols) == 0:
            continue
        base_predictions[cols] = base_predictions[cols].eq(base_predictions[cols].max(axis=1), axis=0).astype(int)
        for c in cols:
            base_predictions['correct'] = np.where(base_predictions[c]==base_groundtruth[c], True, False)
            base_groundtruth = base_groundtruth[base_predictions['correct'] == True]
            base_predictions = base_predictions[base_predictions['correct'] == True]
    print(f"Limiting to correct classifications yields {len(base_groundtruth)} values")
    # save a version of the base_groundtruth file, with all classes, not just output classes
    orig_gt = pd.read_csv(data_args.test_csv)
    orig_gt = orig_gt[orig_gt['Path'].isin(base_groundtruth['Path'].values)]
    orig_gt.to_csv(os.path.join(save_dir,"correct_only_groundtruth.csv"))
    # base_groundtruth.to_csv(os.path.join(save_dir,"correct_only_groundtruth.csv"))
    # set up summary dataframe
    if consistent_triplet:
        print()
        summary_dataframe = pd.DataFrame(columns=[key for key in input_classes] + ['n_samples'] + [f"%{out} (mean)" for out in output_classes] + [f"%{out} (std)" for out in output_classes] + [f"%{out} expected" for out in output_classes])
        for trip in all_possible_triplet_classes:
            trip_class = trip[0]
            idx = len(summary_dataframe)
            summary_dataframe.loc[idx] = [0]*len(summary_dataframe.columns)
            for key, val in input_classes.items():
                for v in val:
                    if v in trip_class:
                        summary_dataframe.at[idx, key] = v
                        if v in output_classes:
                            summary_dataframe.at[idx, f"%{v} expected"] = 100
    else:
        print("Not yet implemented for non consistent triplets")
    summary_dataframe.to_csv(os.path.join(save_dir,"overall_summary.csv"))
    


def run_db_eval(args):
    # TODO: replace placeholders
    input_classes = test_input_classes
    consistent_triplet = True
    n_samples = 50
    output_classes = test_output_classes
    steps = 100
    update_every = 5
    # load arguments
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
    # load model
    ckpt_path = model_args.ckpt_path
    model_args, transform_args\
        = ModelSaver.get_args(cl_model_args=model_args,
                              dataset=data_args.dataset,
                              ckpt_save_dir=Path(ckpt_path).parent,
                              model_uncertainty=model_args.model_uncertainty)
    model_args.moco = args.model_args.moco
    model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                             gpu_ids=args.gpu_ids,
                                             model_args=model_args,
                                             is_training=False)
    predictor = Predictor(model=model, device=args.device, code_dir=args.code_dir)
    # get save_dir
    save_dir = os.path.join("/".join(ckpt_path.split("/")[:-1]), 'decision_boundaries')
    overall_summ_file = os.path.join(save_dir,'overall_summary.csv')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir,"correct_only_groundtruth.csv")):
        print("No trial file found, generating")
        trial_setup(data_args, transform_args, predictor, save_dir, input_classes=input_classes, output_classes=output_classes)
    interaction_subgroups, all_possible_triplet_classes = get_all_combos(input_classes.values(), consistent_triplet=consistent_triplet)
    for trip in all_possible_triplet_classes: # TODO: non consistent triplets
        trip_id = "_".join(["".join(t) for t in trip])
        for i, j in abbreviation_table.items():
            trip_id = trip_id.replace(i,j)
        triplet_save_dir = os.path.join(save_dir, trip_id)
        if not os.path.exists(triplet_save_dir):
            os.mkdir(triplet_save_dir)
        triplet_summary_file = os.path.join(save_dir, f"{trip_id}_summary.csv")
        if os.path.exists(triplet_summary_file):
            triplet_summ = pd.read_csv(triplet_summary_file, index_col=0)
            used_idxs = triplet_summ['idxs'].values.tolist()
        else:
            triplet_summ = pd.DataFrame(columns=['idxs']+[f"%{out}" for out in output_classes])
            used_idxs = []
        if len(triplet_summ) < n_samples:
            # get correct samples from the dataframe
            temp_df = pd.read_csv(os.path.join(save_dir,"correct_only_groundtruth.csv"), index_col=0)
            for t in trip[0]:
                temp_df = temp_df[temp_df[t] == 1]
            for ii in range(len(triplet_summ), n_samples):
                # get idxs 
                current_sample = temp_df.sample(n=3).index.tolist()
                while current_sample in used_idxs:
                    current_sample = temp_df.sample(n=3).index.tolist()
                used_idxs.append(current_sample)
                # time to generate the decision boundary plots!
                inpt_df = pd.read_csv(data_args.test_csv)
                planeloader = get_planeloader(data_args, dataframe=inpt_df, img_idxs=current_sample, subgroups=input_classes, prediction_tasks=model_args.tasks, steps=steps)
                predictions, groundtruth = predictor.predict(planeloader)
                db_results = decision_region_analysis(predictions, 
                                                      planeloader, 
                                                      title_with='both',
                                                      label_with='none',
                                                      classes=output_classes,
                                                      color_dict = {"Yes":"#11721a", "No":"#721a11", "Unknown":"#1385ef"},  # Unknown -> equally predicted for either class
                                                      save_loc=os.path.join(triplet_save_dir, f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}"))
                triplet_summ.loc[ii] = [None]*len(triplet_summ.columns)
                triplet_summ.at[ii, 'idxs'] = current_sample
                for out in output_classes:
                    triplet_summ.at[ii, f"%{out}"] = db_results.at[out, "Percent"]
                triplet_summ.to_csv(triplet_summary_file)
                if (ii+1) %update_every == 0:
                    update_overall_summary(overall_summ_file, triplet_summ, trip[0], input_classes, output_classes) 
        update_overall_summary(overall_summ_file,triplet_summ,trip[0],input_classes, output_classes)
                
def update_overall_summary(overall_summary_file, class_summary_df, triplet_class, input_classes, output_classes):
    df = pd.read_csv(overall_summary_file, index_col=0)
    id_cols = []
    for col in df.columns:
        if col in input_classes:
            id_cols.append(col)
    row_num=None
    for ii, row in df.iterrows():
        row_trip = "_".join(row[id_cols].values)
        if row_trip == "_".join(triplet_class):
            row_num = ii
    if row_num == None:
        print(f"could not find class {triplet_class} in summary file")
        return
    df.at[row_num, "n_samples"] = len(class_summary_df)
    for out in output_classes:
        df.at[row_num, f"%{out} (mean)"] = class_summary_df[f"%{out}"].mean()
        df.at[row_num, f"%{out} (std)"] = class_summary_df[f"%{out}"].std()
    df.to_csv(overall_summary_file)
    
                
if __name__ == '__main__':
    print("Beginning Decision Region Evaluation...")
    test_gt = pd.read_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/7_steps_custom_split/RAND_0/joint_validation.csv")
    parser = TestArgParser()
    run_db_eval(parser.parse_args())
    # trial_setup(parser.parse_args(), input_classes = test_input_classes)
    # decision_boundary_setup(groundtruth=test_gt,
    #                         input_classes = test_input_classes,
    #                         output_classes=['Yes','No'],
    #                         output_dir="/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/db_debug",
    #                         num_samples=20)
    # test_dataset = plane_dataset_2(dataframe=test_gt,
    #                                img_idxs=[13,15,17],
    #                                subgroups=test_input_classes,
    #                                randomize=1,
    #                                random_range=(100,105)
    #                                )
        