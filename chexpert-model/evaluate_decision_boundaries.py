from itertools import product, combinations, combinations_with_replacement
from decision_boundaries import get_planeloader, decision_region_analysis, DecisionBoundaryEvaluator
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
DB_folder = 'DB_uncertainty_test'
num_samples = 500
save_every = 10

def new_run_db_eval(args, db_folder=DB_folder):
    # load arguments
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
    # load model
    ckpt_path = model_args.ckpt_path
    # # adjust DB folder for validation subsets
    if "__" in ckpt_path.split("/")[-1]:
        db_folder = ckpt_path.split("/")[-1].split("__")[0] + "__" + db_folder
    model_args, transform_args = ModelSaver.get_args(cl_model_args=model_args,
                                                     dataset=data_args.dataset,
                                                     ckpt_save_dir=Path(ckpt_path).parent,
                                                     model_uncertainty=model_args.model_uncertainty)
    model_args.moco = args.model_args.moco
    model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                             gpu_ids=args.gpu_ids,
                                             model_args=model_args,
                                             is_training=False)
    predictor = Predictor(model=model, device=args.device, code_dir=args.code_dir)
    # # TRIALS ====================
    DB = DecisionBoundaryEvaluator(db_folder, Path(ckpt_path).parent,  predictor=predictor,
                            input_classes=test_input_classes, output_classes=test_output_classes, 
                            data_args = data_args, model_args=model_args, transform_args=transform_args,
                            n_samples=num_samples, save_every=save_every)
    
# ====================================================================================
#                              Deprecated Code
# ====================================================================================

# edit the triplet classes to use - set to None to use all combos from test_input_classes
# test_input_triplet = [[("M","White","Yes"),("M","White","Yes"),("M","White","Yes")],
#                       [("M","White","No"),("M","White","No"),("M","White","No")]]
# test_input_triplet = [[("M","Black_or_African_American","Yes"),("M","Black_or_African_American","Yes"),("M","Black_or_African_American","Yes")],
#                       [("M","Black_or_African_American","No"),("M","Black_or_African_American","No"),("M","Black_or_African_American","No")]]
# test_input_triplet = [[("F","White","Yes"),("F","White","Yes"),("F","White","Yes")],
#                       [("F","White","No"),("F","White","No"),("F","White","No")]]
# test_input_triplet = [[("F","Black_or_African_American","Yes"),("F","Black_or_African_American","Yes"),("F","Black_or_African_American","Yes")],
#                       [("F","Black_or_African_American","No"),("F","Black_or_African_American","No"),("F","Black_or_African_American","No")]]                     

# test_input_triplet = [[("M","Black_or_African_American","Yes"),("M","Black_or_African_American","Yes"),("M","Black_or_African_American","Yes")],
#                       [("M","Black_or_African_American","No"),("M","Black_or_African_American","No"),("M","Black_or_African_American","No")],
#                       [("M","White","Yes"),("M","White","Yes"),("M","White","Yes")],
#                       [("M","White","No"),("M","White","No"),("M","White","No")]]

# test_input_triplet = [[("F","White","Yes"),("F","White","Yes"),("F","White","Yes")],
#                       [("F","White","No"),("F","White","No"),("F","White","No")],
#                       [("F","Black_or_African_American","Yes"),("F","Black_or_African_American","Yes"),("F","Black_or_African_American","Yes")],
#                       [("F","Black_or_African_American","No"),("F","Black_or_African_American","No"),("F","Black_or_African_American","No")]]
test_input_triplet = None
# test_input_triplet = [[("M","White","Yes"),("M","White","Yes"),("M","White","Yes")]]
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
inv_abbreviation_table = {value:key for key,value in abbreviation_table.items()}

# db_folder = 'decision_boundaries_exps'
# db_folder = "DB_debug"
db_folder = 'decision_boundaries_all_scores'

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

def get_pid_combos(pids, n_pid_use):
    out_pids = []
    for r in range(n_pid_use):
        r_pids = [p for p in pids]
        for n in range(int(len(pids)/3)):
            item = sample(r_pids, 3)
            while set(item) in out_pids: # don't allow the same exact combo of patients multiple times
                item = sample(r_pids,3)
            for i in item:
                r_pids.remove(i)
            out_pids.append(set(item))
        # print("r: ", r)
        # print(len(out_pids))
    return [list(p) for p in out_pids]

def trial_setup(data_args, transform_args,  predictor, save_dir, input_classes=test_input_classes, output_classes=test_output_classes, consistent_triplet=True, num_samples=10):
    # determine the possible combinations
    interaction_subgroups, all_possible_triplet_classes = get_all_combos(input_classes.values(), consistent_triplet=consistent_triplet)
    # load groundtruth, limit to only correct classifications
    orig_gt = pd.read_csv(data_args.test_csv)
    loader = get_loader(phase = data_args.phase,
                        data_args = data_args,
                        transform_args=transform_args,
                        is_training=False,
                        return_info_dict=True)
    base_predictions, base_groundtruth, base_paths = predictor.predict(loader)
    base_predictions['patient_id'] = base_predictions.Path.map(orig_gt.set_index("Path").patient_id)
    
    # base_predictions, base_groundtruth = predictor.predict(loader, by_patient=False)
    print(f"Base predictions has {len(base_predictions)} values ({len(base_predictions['patient_id'].unique())} patients)")
    prediction_output = orig_gt.copy()
    for key, vals in input_classes.items():
        cols = [v for v in vals if v in output_classes]
        if len(cols) == 0:
            continue
        for c in cols:
            prediction_output[f"{c} raw prediction"] = prediction_output.Path.map(base_predictions.set_index("Path")[c])
        base_predictions[cols] = base_predictions[cols].eq(base_predictions[cols].max(axis=1), axis=0).astype(int)
        for c in cols:
            prediction_output[f"{c} prediction"] = prediction_output.Path.map(base_predictions.set_index("Path")[c])
        for c in cols:
            prediction_output['correct'] = np.where(prediction_output[f"{c} prediction"] == prediction_output[c], True, False)
            base_predictions['correct'] = np.where(base_predictions[c]==base_groundtruth[c], True, False)
    prediction_output.to_csv(os.path.join(save_dir, "complete_prediction_output.csv"), index='False')
    # base_predictions.to_csv(os.path.join(save_dir,"predictions.csv"))
    # set up summary dataframe
    if consistent_triplet:
        summary_dataframe = pd.DataFrame(columns=[key for key in input_classes] +  ['n_samples'] + [f"%{out} (mean)" for out in output_classes] + [f"%{out} (std)" for out in output_classes] + [f"%{out} expected" for out in output_classes])
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
    


def run_db_eval(args, db_folder=db_folder):
    # TODO: replace placeholders
    input_classes = test_input_classes
    consistent_triplet = True
    n_samples = 500
    # n_samples = 5
    # n_samples = 10
    output_classes = test_output_classes
    steps = 100
    update_every = 10
    plot_shape = 'triangle'
    # plot_shape = 'rectangle'
    save_every = 10
    generate_plot = False
    plot_colors = {"Yes":"#11721a", "No":"#721a11", "Unknown":"#1385ef"}
    # n_patient_repeat = 15
    triplet_classes = test_input_triplet
    perturb_exp = False
    subset = 'all' # all, correct_only, or incorrect_only
    if perturb_exp:
        print("Running in experimental mode")
    # triplet_classes=None
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
    # check for validation subset
    if "__" in ckpt_path.split("/")[-1]:
        val_sub = ckpt_path.split("/")[-1].split("__")[0]
        db_folder = val_sub + "_" + db_folder
    save_dir = os.path.join("/".join(ckpt_path.split("/")[:-1]), db_folder)
    print("\nSAVE DIR: ", save_dir)
    overall_summ_file = os.path.join(save_dir,'overall_summary.csv')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir,"complete_prediction_output.csv")):
        print("No trial file found, generating")
        trial_setup(data_args, transform_args, predictor, save_dir, input_classes=input_classes, output_classes=output_classes)
    if triplet_classes == None: # get the combinations of triplets if triplets not provided!
        interaction_subgroups, all_possible_triplet_classes = get_all_combos(input_classes.values(), consistent_triplet=consistent_triplet)
    elif type(triplet_classes) == str: # for input in form "MWP,FBN,FWP,MWN,..." -> still only for consistent triplets
        t_classes = triplet_classes.split(",")
        all_possible_triplet_classes = []
        for t in t_classes:
            x = list(t)
            for ii, y in enumerate(x):
                if y in inv_abbreviation_table:
                    x[ii] = inv_abbreviation_table[y]
            all_possible_triplet_classes.append([tuple(x)]*3)
    else:
        all_possible_triplet_classes = triplet_classes
    
    for trip in all_possible_triplet_classes: # TODO: non consistent triplets
    
        trip_id = "_".join(["".join(t) for t in trip])
        for i, j in abbreviation_table.items():
            trip_id = trip_id.replace(i,j)
        triplet_save_dir = os.path.join(save_dir, trip_id)
        triplet_array_file = os.path.join(save_dir, f"{trip_id}.npz")
        triplet_idx_file = os.path.join(save_dir, f'{trip_id}_idxs.json')
        if not os.path.exists(triplet_idx_file): # restricted by n_patient_repeat and the total number of patients w/ a correctly predicted img
            # get triplet idxs
            # get the patient_ids for the triplet class
            temp_df = pd.read_csv(os.path.join(save_dir,"complete_prediction_output.csv"), index_col=0)
            if subset == 'correct_only':
                temp_df = temp_df[temp_df['correct'] == True]
            elif subset == 'incorrect_only':
                temp_df = temp_df[temp_df['correct'] == False]
            for t in trip[0]:
                temp_df = temp_df[temp_df[t] == 1]
            n_patient_repeat = int((n_samples/(len(temp_df['patient_id'].unique())/3))+1)
            print(f"repeating each of {len(temp_df['patient_id'].unique())} patients {n_patient_repeat} times to get {n_samples} samples")
            # get patient_ids 
            pids = list(temp_df['patient_id'].unique())
            triplet_idxs = get_pid_combos(pids, n_patient_repeat)
            idx_df = pd.DataFrame(triplet_idxs, columns=['pid0','pid1','pid2'])
            idx_df.to_json(triplet_idx_file)
        else:
            idx_df = pd.read_json(triplet_idx_file)
        if not os.path.exists(triplet_save_dir) and generate_plot:
            os.mkdir(triplet_save_dir)
        triplet_summary_file = os.path.join(save_dir, f"{trip_id}_summary.csv")
        all_potential_idxs_file = os.path.join(save_dir, f"{trip_id}_idxs.json")
        if os.path.exists(triplet_summary_file): # resuming trial
            triplet_summ = pd.read_csv(triplet_summary_file, index_col=0)
            triplet_idxs = pd.read_json(all_potential_idxs_file)
            # used_idxs = triplet_summ['idxs'].values.tolist()
            DB_arrays = np.load(triplet_array_file, allow_pickle=True)
            DB_arrays = dict(DB_arrays)
        else:
            if not perturb_exp:
                triplet_summ = pd.DataFrame(columns=['idxs']+[f"%{out}" for out in output_classes])
            else:
                triplet_summ = pd.DataFrame(columns=['idxs','mode']+[f"%{out}" for out in output_classes])
            DB_arrays = {}
        if len(triplet_summ) < n_samples:
            temp_df = pd.read_csv(os.path.join(save_dir,"complete_prediction_output.csv"), index_col=0)
            if subset == 'correct_only':
                temp_df = temp_df[temp_df['correct'] == True]
            elif subset == 'incorrect_only':
                temp_df = temp_df[temp_df['correct'] == False]
            for t in trip[0]:
                temp_df = temp_df[temp_df[t] == 1]
            for ii in range(len(triplet_summ), n_samples):
                # get idxs  -> choose a random image from each of the specified patients
                current_sample = []
                for pid in ['pid0','pid1','pid2']:
                    current_sample.append(temp_df[temp_df['patient_id'] == idx_df.at[ii,pid]].sample(n=1).index.tolist()[0])
                # time to generate the decision boundary plots!
                # DEBUG
                # current_sample = [4307,2375, 778]
                inpt_df = pd.read_csv(data_args.test_csv)
                if not perturb_exp:
                    planeloader = get_planeloader(data_args, dataframe=inpt_df, img_idxs=current_sample, subgroups=input_classes, prediction_tasks=model_args.tasks, steps=steps, shape=plot_shape)
                    predictions, groundtruth = predictor.predict(planeloader)
                    db_results, db_array = decision_region_analysis(predictions, 
                                                        planeloader, 
                                                        title_with='both',
                                                        label_with='none',
                                                        classes=output_classes,
                                                        generate_plot=generate_plot,
                                                        color_dict = plot_colors,
                                                        save_loc=os.path.join(triplet_save_dir, f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}"))
                    DB_arrays[f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}"] = db_array
                    triplet_summ.loc[ii] = [None]*len(triplet_summ.columns)
                    triplet_summ.at[ii, 'idxs'] = current_sample
                    for out in output_classes:
                        triplet_summ.at[ii, f"%{out}"] = db_results.at[out, "Percent"]
                    
                elif perturb_exp: # perturbation experiments
                    # normal mode
                    planeloader = get_planeloader(data_args, save_images=save_dir, dataframe=inpt_df, img_idxs=current_sample, subgroups=input_classes, prediction_tasks=model_args.tasks, steps=steps, shape=plot_shape)
                    predictions, groundtruth = predictor.predict(planeloader)
                    db_res, db_array = decision_region_analysis(predictions,
                                                                    planeloader,
                                                                    title_with='both',
                                                                    label_with='none',
                                                                    classes=output_classes,
                                                                    generate_plot=generate_plot,
                                                                    color_dict=plot_colors,
                                                                    save_loc=os.path.join(triplet_save_dir, f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_base")) 
                    DB_arrays[f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_base"] = db_array
                    ts_idx = len(triplet_summ)
                    triplet_summ.loc[ts_idx] = [None]*len(triplet_summ.columns)
                    triplet_summ.at[ts_idx,'idxs'] = current_sample
                    triplet_summ.at[ts_idx,'mode'] = 'base'
                    for out in output_classes:
                        triplet_summ.at[ts_idx, f"%{out}"] = db_res.at[out, "Percent"]
                    for rn in range(1,4):
                        # shuffle
                        planeloader = get_planeloader(data_args, save_images=save_dir,dataframe=inpt_df, img_idxs=current_sample, subgroups=input_classes, prediction_tasks=model_args.tasks, steps=steps, shape=plot_shape, shuffle=rn)
                        predictions, groundtruth = predictor.predict(planeloader)
                        db_res, db_array = decision_region_analysis(predictions,
                                                                        planeloader,
                                                                        title_with='both',
                                                                        label_with='none',
                                                                        classes=output_classes,
                                                                        generate_plot=generate_plot,
                                                                        color_dict=plot_colors,
                                                                        save_loc=os.path.join(triplet_save_dir, f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_shuffle_{rn}")) 
                        DB_arrays[f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_shuffle_{rn}"] = db_array
                        ts_idx = len(triplet_summ)
                        triplet_summ.loc[ts_idx] = [None]*len(triplet_summ.columns)
                        triplet_summ.at[ts_idx,'idxs'] = current_sample
                        triplet_summ.at[ts_idx,'mode'] = f'shuffle_{rn}'
                        for out in output_classes:
                            triplet_summ.at[ts_idx, f"%{out}"] = db_res.at[out, "Percent"]
                        # random
                        for random_range_width in [256,128,64]:
                            for r_start in range(0,256, random_range_width):
                                r_end = r_start + (random_range_width-1)
                                planeloader = get_planeloader(data_args,save_images=save_dir, dataframe=inpt_df, img_idxs=current_sample, subgroups=input_classes, prediction_tasks=model_args.tasks, steps=steps, shape=plot_shape, randomize=rn, random_range=(r_start,r_end))
                                predictions, groundtruth = predictor.predict(planeloader)
                                db_res, db_array = decision_region_analysis(predictions,
                                                                                planeloader,
                                                                                title_with='both',
                                                                                label_with='none',
                                                                                classes=output_classes,
                                                                                generate_plot=generate_plot,
                                                                                color_dict=plot_colors,
                                                                                save_loc=os.path.join(triplet_save_dir, f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_random_{rn}_({r_start}_{r_end})")) 
                                DB_arrays[f"{ii+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_random_{rn}_({r_start}_{r_end})"] = db_array
                                ts_idx = len(triplet_summ)
                                triplet_summ.loc[ts_idx] = [None]*len(triplet_summ.columns)
                                triplet_summ.at[ts_idx,'idxs'] = current_sample
                                triplet_summ.at[ts_idx,'mode'] = f'random_{rn}_({r_start}_{r_end})'
                                for out in output_classes:
                                    triplet_summ.at[ts_idx, f"%{out}"] = db_res.at[out, "Percent"]
                if (ii+1) % save_every == 0:
                    triplet_summ.to_csv(triplet_summary_file)
                    np.savez_compressed(triplet_array_file, **DB_arrays)
                if (ii+1) % update_every == 0:
                    update_overall_summary(overall_summ_file, triplet_summ, trip[0], input_classes, output_classes)

                # return # DEBUG
            triplet_summ.to_csv(triplet_summary_file) # save at the end of DB for indiviual triplet
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
    parser = TestArgParser()
    new_run_db_eval(parser.parse_args())
        