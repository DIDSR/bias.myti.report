import os
import pandas as pd
from args import TestArgParser
from constants import *
from saver import ModelSaver
from data import get_loader
from predict import Predictor
import torch

'''
    generates prediction and groundtruth files for specified model checkpoint.
'''
def predict(args):
    # argument management
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args
    # load model
    ckpt_path = model_args.ckpt_path
    ckpt_save_dir = Path(ckpt_path).parent
    model_uncertainty = model_args.model_uncertainty
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
    # get valid loader
    # data_args.tasks = args.tasks
    data_loader = get_loader(phase=data_args.phase,
                            data_args=data_args,
                            transform_args=transform_args,
                            is_training=False,
                            return_info_dict=True,
                            logger=None)
    # get predictions 
    
    predictor = Predictor(model=model, device=args.device, code_dir=args.code_dir)
    predictions, groundtruth, paths = predictor.predict(data_loader, by_patient=args.by_patient)

    # predictions.to_csv(args.prediction_save_file, index=False)
    # groundtruth.to_csv(args.prediction_save_file.replace("predictions", "groundtruth"), index=False)
    predictions.to_csv(args.prediction_save_file, index=True)
    groundtruth.to_csv(args.prediction_save_file.replace("predictions", "groundtruth"), index=True)

if __name__ == '__main__':
    parser = TestArgParser()
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser.parser.add_argument("--prediction_save_file", required=True)
    parser.parser.add_argument("--tasks")
    args = parser.parse_args()
    predict(args)