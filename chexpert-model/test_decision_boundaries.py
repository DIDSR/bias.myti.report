import csv
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
import torch


def test(args,
         db_steps = 100,
         save_result_images = False,
         selection_mode = 'index',
         plane_samples = [323, 347, 79],
         csv_input ="/gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/RAND_1/ts__20220801_summary_table__MIDRC_RICORD_1C.csv",
         plot_save_loc = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/SPIE2023_figures/abstract_decision_boundary_example.png"):
    # # ========== Plane Generation Settings ==========
    #db_steps =  200 # number of steps for linspace, will generate db_steps^2 images
    #save_result_images = False
    #selection_mode = 'class'
    #plane_samples =['FDX','MCR','MDX']
    #csv_input = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/TCIA_1C_train.csv"
    # # ========== Decision Boundary Plot Settings ==========
    synthetic_predictions = False # if true, will assign fake prediction results
    #plot_save_loc = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/decision_boundaries/TCIA_1C_test.png"
    classes = ['Female-CR', 'Female-DX','Male-CR', 'Male-DX']
    plot_mode = 'overlap'
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
        print(f"Model checkpoint info: {ckpt_info}")
        # Instantiate the Predictor class for obtaining model predictions.
        predictor = Predictor(model=model, device=args.device)
        # Get phase loader object.
        return_info_dict = False
        # # ===================== Predict on validation set =========
        '''
        data_args.csv_dev = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/TCIA_1C_valid.csv"
        data_args.metric_name = 'custom-AUROC'
        data_args.custom_tasks = 'custom-tasks'
        valid_loader = get_loader(phase = "valid", data_args=data_args, transform_args=transform_args,
        is_training=False, return_info_dict=False, logger=logger)
        predictions, gt = predictor.predict(valid_loader)
        print("validation set predictions (first 10):")
        print(predictions.head(10))'''
        # validation predications indicate that the model is being imported properly
        
        # # ===================== adding planeloader ================
        print("creating planeloader...")
        loader = get_planeloader(data_args=data_args,
                                 csv_input=csv_input,
                                 steps=db_steps,
                                 selection_mode=selection_mode,
                                 samples=plane_samples,
                                 data_mode = 'normal',
                                 save_result_images=save_result_images)

        print(f"found {len(loader.dataset)} images in the dataset")
        
        # Obtain model predictions.
        if return_info_dict:
            predictions, groundtruth, paths = predictor.predict(loader)
        else:
            predictions, groundtruth = predictor.predict(loader)
           
        # ================== Decision Boundaries ==================
        plot_decision_boundaries(predictions = predictions,
                                planeloader=loader,
                                synthetic_predictions=synthetic_predictions,
                                classes=classes,
                                plot_mode=plot_mode,
                                save_loc=plot_save_loc)
        print('Done')
        return


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    print("Starting...")
    test(parser.parse_args())