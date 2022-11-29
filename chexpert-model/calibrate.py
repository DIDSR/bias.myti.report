from args import TestArgParser
from constants import *
from saver import ModelSaver
from data import get_loader
from calibration_utils import *
from predict import Predictor

def calibrate_model(args):
    print("Beginning model calibration")
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
    valid_loader = get_loader(phase=data_args.phase,
                              data_args=data_args,
                              transform_args=transform_args,
                              is_training=False,
                              return_info_dict=True,
                              logger=None)
    # get predictions 
    predictor = Predictor(model=model, device=args.device, code_dir=args.code_dir)
    predictions, groundtruth, paths = predictor.predict(valid_loader)
    combined = groundtruth.copy()
    
    pred_cols = []
    label_cols = []
    for t in data_args.custom_tasks.split(","):
        combined[f'{t} gt'] = combined['Path'].map(predictions.set_index('Path')[t])
        pred_cols.append(f"{t} gt")
        label_cols.append(t)
    preds = torch.tensor(combined[pred_cols].values)
    labels = torch.tensor(combined[label_cols].values)
    # run calibration
    calModel = CalibratedModel(model=None, calibration_mode=args.calibration_mode)
    calModel.set_temperature(valid_results=[preds, labels])
    calModel.save(ckpt_path.replace("best.pth.tar", "temperature_scaling.pt"))

    

if __name__ == "__main__":
    parser = TestArgParser()
    parser.parser.add_argument("--calibration_mode", choices=['temperature'])
    parser.parser.add_argument("--tasks", dest='data_args.custom_tasks') # not sure why this isn't importing from model properly
    calibrate_model(parser.parse_args())
    print()