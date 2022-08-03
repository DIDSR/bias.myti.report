from args import TestArgParser
from decision_boundaries import get_sample_list, get_planeloader, plot_decision_boundaries
from test_decision_boundaries import test
from logger import Logger
from predict import Predictor, EnsemblePredictor
from constants import *
from saver import ModelSaver
import pandas as pd
import torch
import itertools
import os
import random
import csv
# subclass combinations:
# already complete/in progress (1000 samples):
    # MDX, MDX, MCR
    # MDX, MDX, FDX
    # FDX, FDX, FCR
# # Ravi's runs
# # # MCR, MCR, FCR - Completed 
# # # 'FCR', 'FCR', 'MCR' - running FCR_FCR_MCR - done
# # # 'FDX', 'FDX', 'MDX' - running FDX_FDX_MDX - done
# # # 'FCR', 'FCR', 'FDX' - running FCR_FCR_FDX - 
# # # 'MCR', 'MCR', 'MDX' - running MCR_MCR_MDX - restarted
# # # 'MDX', 'MCR', 'FCR' - running - MDX_MCR_FCR
# # # 'MDX', 'MDX', 'MDX' - running - MDX_MDX_MDX
# # # 'MCR', 'MCR', 'MCR' - running - MCR_MCR_MCR
# # # 'FDX', 'FDX', 'FDX' - running - FDX_FDX_FDX
# # # 'FCR', 'FCR', 'FCR' - running - FCR_FCR_FCR
# not yet complete:
    # 
    # 
    # 
    # 
    # ------
    # MDX, MCR, FDX - Alexis
    # MDX, MCR, FCR
    # FDX, FCR, MDX
    # FDX, FCR, MCR
    # 
def test_bulk(args,
              train_or_valid = 'train',
              subclasses =['FCR', 'FCR', 'FCR'],
              n_samples = 1000,
              test_name = "FCR_FCR_FCR_100_steps_v1",
              db_steps=100,
              output_folder = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/decision_boundaries",
              overwrite=True,
              point_size=10):
    print(f"===== Beginning test {test_name} ===================================")
    print(f"subclasses: {subclasses}")
    print(f"number of samples: {n_samples}")
    # # ===== Config =======================================
    # select csv_file
    if train_or_valid == 'train':
        csv_file = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/TCIA_1C_train.csv"
    elif train_or_valid == 'valid':
        csv_file = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/TCIA_1C_valid.csv"
    # create folder and summary csv file
    save_path = os.path.join(output_folder, test_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elif not overwrite:
        print("Error: output directory already exists")
        return
    summary_csv = os.path.join(save_path, 'summary.csv')
    with open(summary_csv, 'a') as file:
        writer = csv.DictWriter(file,
                                fieldnames=["image 1 class", "image 2 class", "image 3 class",
                                "image 1 index", "image 2 index", "image 3 index",
                                "F-CR occurances", "F-DX occurances", "M-CR occurances", "M-DX occurances",
                                "F-CR percent", "F-DX percent", "M-CR percent", "M-DX percent"])
        writer.writeheader()
    # Arguments
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
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
    # save test information
    summ_file = os.path.join(save_path, "test_information.txt")
    with open(summ_file, 'w') as file:
        file.write(f"""Subclasses: {subclasses} \n
        Model checkpoint information: {ckpt_info} \n
        Model checkpoint path: {ckpt_path}""")
    # # ===== Find Sample Combinations =====================
    # find available samples based on subclasses
    df = pd.read_csv(csv_file)
    possible_samples = get_sample_list(subclasses, df)
    all_combinations = list(itertools.product(*possible_samples))
    # remove all combinations that use the same image twice 
    combinations = [item for item in all_combinations if len(dict.fromkeys(item)) == 3]
    print(f"Found {len(combinations)} possible combinations of the subclasses {subclasses}")
    
    # # ===== Begin Generating Plots ========================
    for i in range(n_samples):
        print(f"---------- Plot {i+1}/{n_samples} ----------")
        # select the set of samples
        idx = random.randint(0, len(combinations))
        sample_idxs = combinations[idx]
        combinations.pop(idx)
        plot_name = f"{i+1}__({sample_idxs[0]}_{sample_idxs[1]}_{sample_idxs[2]}).png"
        # create planeloader
        loader = get_planeloader(data_args=data_args,
                                 csv_input=csv_file,
                                 steps=db_steps,
                                 selection_mode='index',
                                 samples=sample_idxs,
                                 save_result_images=False)

        # get predictions
        predictions, groundtruth = predictor.predict(loader)

        summ_df = plot_decision_boundaries(predictions,
                                        loader,
                                        classes = ['Female-CR', 'Female-DX','Male-CR', 'Male-DX'],
                                        save_loc=os.path.join(save_path, plot_name),
                                        point_size=point_size)
        print(summ_df)
        summ_list = []
        summ_list += [subclass for subclass in subclasses]
        summ_list += [idx for idx in sample_idxs]
        subgroups = ['F-CR', 'F-DX', 'M-CR', 'M-DX']
        summ_list += [summ_df.iloc[i]['occurances'] for i in range(len(subgroups))]
        summ_list += [summ_df.iloc[i]['percent'] for i in range(len(subgroups))]
        
        with open(summary_csv, 'a', newline='') as file:
            file_writer = csv.writer(file)
            file_writer.writerow(summ_list)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    test_bulk(parser.parse_args())
