from decision_boundaries import get_sample_list, get_planeloader, plot_decision_boundaries
from args import TestArgParser
import pandas as pd
import argparse
import itertools
import json
import csv
import os
import random
from logger import Logger
from predict import Predictor, EnsemblePredictor
from constants import *
from saver import ModelSaver


def conduct_trial(args,
                  save_loc="/gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/decision_boundaries",
                  trial_name = None,
                  input_csv = None):
    print('Loading Model...')
    # Arguments
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
    # FOR SPIE2023 runs =============================================================================
    base_folder = model_args.ckpt_path.replace("/full_MIDRC_RICORD_1C/best.pth.tar", '')
    trial_name = base_folder.split("/")[-1]
    input_csv = os.path.join(base_folder, 'tr__20220801_summary_table__MIDRC_RICORD_1C.csv')
    overall_summ_file = os.path.join(save_loc, f"{trial_name}_summary.csv")
    n_to_update=10
    # ===============================================================================================
    print(f"=========== {trial_name} ==========")
    print(f"input csv: {input_csv} \n")
    print(f"model checkpoint path: {model_args.ckpt_path} \n")
    
    # get logger
    logger = Logger(logger_args.log_path,
                    logger_args.save_dir,
                    logger_args.results_dir)
    # load model from checkpoint
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
    model_args.moco = args.model_args.moco
    model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                                gpu_ids=args.gpu_ids,
                                                model_args=model_args,
                                                is_training=False)
    print(f"Model checkpoint info: {ckpt_info}")
    # create predictor
    predictor = Predictor(model=model, device=args.device)
    # check if the trial already exists, if not, create a new trial
    trial_folder = os.path.join(save_loc, trial_name)
    if not os.path.exists(trial_folder):
        print('Starting new trial...')
        trial_setup(trial_name=trial_name, save_loc=save_loc, summary_csv=overall_summ_file)
    else:
        print("Resuming trial...")

    print('Loading trial information ...')
    # load trial information
    json_file = os.path.join(trial_folder, 'tracking_info.json')
    with open(json_file, 'r') as infile:
        tracking_info = json.load(infile)
    for x in range(len(tracking_info['combinations'])):
        if tracking_info['combinations'][str(x)]['current_samples'] == tracking_info['max_samples']:
            # skip if complete
            print(f"Combination {x+1} {tracking_info['combinations'][str(x)]['subgroups']} complete")
            continue
        print(f"===== Combination {x+1}/{len(tracking_info['combinations'])} {tracking_info['combinations'][str(x)]['subgroups']} =====")
        # file management
        s = [str(idx) for idx in tracking_info['combinations'][str(x)]['subgroups']]
        combo_name = f'{s[0]}_{s[1]}_{s[2]}'
        plot_loc = os.path.join(tracking_info['save_loc'], combo_name)
        if not os.path.exists(plot_loc):
            os.makedirs(plot_loc)
        combo_summary_csv = os.path.join(tracking_info['save_loc'], f'{combo_name}_summary.csv')
        if not os.path.exists(combo_summary_csv):
            with open(combo_summary_csv, 'w') as file:
                writer = csv.DictWriter(file, fieldnames=['image classes', 'image indexs',
                        "F-CR occurances", "F-DX occurances", "M-CR occurances", "M-DX occurances",
                        "F-CR percent", "F-DX percent", "M-CR percent", "M-DX percent"])
                writer.writeheader()
            used_indexs = []
        else:
            # get the already used combinations
            df = pd.read_csv(combo_summary_csv)
            used_indexs = df['image indexs'].tolist()
        # find all possible combinations from the input
        in_df = pd.read_csv(input_csv)
        possible_samples = get_sample_list(tracking_info['combinations'][str(x)]['subgroups'], in_df)
        all_combinations = list(itertools.product(*possible_samples))
        # remove all combinations that use the same image twice, or already used
        combinations = [item for item in all_combinations if len(dict.fromkeys(item)) == 3]
        for idx in used_indexs:
            if idx in combinations:
                combinations.remove(idx)
        for i in range(len(used_indexs), tracking_info['max_samples']):
            print(f"---------- plot {i+1}/{tracking_info['max_samples']} ----------")
            # get indexs
            idx = random.randint(0, len(combinations))
            sample_idxs = combinations[idx]
            combinations.pop(idx)
            plot_name = f"{i+1}__({sample_idxs[0]}_{sample_idxs[1]}_{sample_idxs[2]}).png"
            # create planeloader
            loader = get_planeloader(data_args=data_args,
                                 csv_input=input_csv,
                                 steps=tracking_info['steps'],
                                 selection_mode='index',
                                 samples=sample_idxs,
                                 save_result_images=False)

            # get predictions
            predictions, groundtruth = predictor.predict(loader)
            summ_df = plot_decision_boundaries(predictions,
                                        loader,
                                        classes = ['Female-CR', 'Female-DX','Male-CR', 'Male-DX'],
                                        save_loc=os.path.join(plot_loc, plot_name),
                                        point_size=10)
            print(summ_df)
            summ_list = []
            summ_list += [tracking_info['combinations'][str(x)]['subgroups']]
            summ_list += [sample_idxs]
            subgroups = ['F-CR', 'F-DX', 'M-CR', 'M-DX']
            summ_list += [summ_df.iloc[i]['occurances'] for i in range(len(subgroups))]
            summ_list += [summ_df.iloc[i]['percent'] for i in range(len(subgroups))]
            
            with open(combo_summary_csv, 'a', newline='') as file:
                file_writer = csv.writer(file)
                file_writer.writerow(summ_list)
            tracking_info['combinations'][str(x)]['current_samples'] += 1
            with open(json_file, 'w') as file:
                json.dump(tracking_info, file)
            if (i+1)%n_to_update == 0:
                print(f"=== Update {i+1} plots ===")
                get_update(pd.read_csv(combo_summary_csv))
            
        # get mean and std from summary information
        info_df = pd.read_csv(combo_summary_csv)
        summ_info = [tracking_info['combinations'][str(x)]['subgroups']]
        for item in ['F-CR percent', 'F-DX percent', 'M-CR percent', 'M-DX percent']:
            mn = info_df.describe()[item]['mean']
            sd = info_df.describe()[item]['std']
            summ_info.append(f"{mn:.3f} ({sd:.3f})")
        # save to overall summary file
        with open(overall_summ_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(summ_info)
    return
def get_update(combo_df):
    for item in ['F-CR percent', 'F-DX percent', 'M-CR percent', 'M-DX percent']:
        mn = combo_df.describe()[item]['mean']
        sd = combo_df.describe()[item]['std']
        print(f"{item}: {mn:.3f} ({sd:.3f})")

def trial_setup(trial_name, save_loc,
                subgroups=[('MDX','MDX','MDX'),('MCR','MCR','MCR'),
                           ('FDX', 'FDX', 'FDX'), ('FCR','FCR','FCR')],
                samples=250,
                steps=100,
                classes=['FCR', 'FDX', 'MCR', 'MDX'],
                summary_csv = None
                ):
    '''
    Creates the necessary files and folders
    '''
    print('setting up trial...')
    if not subgroups:
        subgroups = []
        # create subgroups from all possible combinations of classes
        combinations = itertools.product(classes,repeat=3)
        for combo in combinations:
            subgroups.append(combo)
    # create trial folder
    trial_folder = os.path.join(save_loc, trial_name)
    if os.path.exists(trial_folder):
        print(f"Folder with the name {trial_name} already exists. returning to avoid overwriting data.")
        return
    else:
        os.makedirs(trial_folder)
    # create summary csv
    if not summary_csv:
        summary_csv = os.path.join(trial_folder, 'summary.csv')
    
    with open(summary_csv,'w', newline='') as file:
        # create header
        writer = csv.DictWriter(file,
                                fieldnames=['classes']+[cls for cls in classes])
        writer.writeheader()
    # create tracking json file
    tracking_info = {'combinations':{},
                     'steps':steps,
                     'save_loc':trial_folder,
                     'max_samples':samples}
    for i in range(len(subgroups)):
        tracking_info['combinations'][i] = {'subgroups':subgroups[i],
                                            'current_samples':0}
    json_file = os.path.join(trial_folder, 'tracking_info.json')
    with open(json_file, 'w') as file:
        json.dump(tracking_info, file)


if __name__ == '__main__':
    parser = TestArgParser()
    conduct_trial(parser.parse_args())