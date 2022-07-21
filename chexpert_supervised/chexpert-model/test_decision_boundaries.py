from decision_boundaries import get_planeloader, plot_decision_boundaries
from logger import Logger
from predict import Predictor, EnsemblePredictor
from saver import ModelSaver
from data import get_loader
from eval import Evaluator
from constants import *
from scripts.get_cams import save_grad_cams
from dataset import TASK_SEQUENCES
from args import TestArgParser
import argparse
import torch

def test(args):
    # # ========== Decision Boundary Plot Settings ==========
    db_steps = 100 # number of steps for linspace, will generate db_steps^2 images
    synthetic_predictions = True # if true, will assign fake prediction results
    plot_save_loc = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/decision_boundaries/test.png"
    # # =====================================================
    model_args = args.model_args
    data_args = args.data_args
    logger_args = args.logger_args

    # import pdb; pdb.set_trace()

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

        # Instantiate the Predictor class for obtaining model predictions.
        predictor = Predictor(model=model, device=args.device)
        # Get phase loader object.
        return_info_dict = False
        # # ===================== adding planeloader ================
        print("creating planeloader...")
        loader = get_planeloader(data_args=data_args)
        print(f"found {len(loader.dataset)} images in the dataset")

        # Obtain model predictions.
        if return_info_dict:
            predictions, groundtruth, paths = predictor.predict(loader)
        else:
            predictions, groundtruth = predictor.predict(loader)
        # ================== Decision Boundaries ==================
        plot_decision_boundaries(predictions, loader, loader.dataset.base_labels, synthetic_predictions, plot_save_loc)
        print('Done')
        return


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    print("Starting...")
    test(parser.parse_args())