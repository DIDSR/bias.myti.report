"""Entry-point script to train models."""
import torch
import torch.nn as nn
import pandas as pd


import models
from args import TrainArgParser
from logger import Logger
from saver import ModelSaver
from predict import Predictor
from data import get_loader
from eval import Evaluator
from optim import Optimizer
from constants import *
import json
from datetime import datetime
# from torchsummaryX import summary


def train(args):
    import os # Why does os only import properly if in the function?
    """Run model training."""

    print("Start Training ...")
    # set random state, if specified
    if args.random_state is not None:
        torch.manual_seed(args.random_state)

    # Get nested namespaces.
    model_args = args.model_args
    logger_args = args.logger_args
    optim_args = args.optim_args
    data_args = args.data_args
    transform_args = args.transform_args
    # # Changes made for continual_learning_evaluation =============================
    
    # New tracking method 10/31/2022
    partition_tracking_fp = os.path.join(("/").join(str(logger_args.save_dir).split("/")[:-1]), 'tracking.log')
    model_tracking_fp = os.path.join(logger_args.save_dir, 'model_tracking.log')
    # generate model tracking info
    # get tasks
    if data_args.custom_tasks:
        if data_args.custom_tasks in NamedTasks:
            tasks = NamedTasks[data_args.custom_tasks]
        else:
            tasks = data_args.custom_tasks.split(",")
    tracking_info = {
        'Experiment Name':logger_args.experiment_name,
        'Base Weights':args.model_args.ckpt_path,
        'max_epochs':args.optim_args.num_epochs,
        'tasks':tasks,
        'random_state':args.random_state,
        'step':int(data_args.csv.replace(".csv","").split("_")[-1]),
        'training started':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training completed':None,
        'best iter info':{
            'loss threshold':args.selection_args.loss_threshold,
            'loss std threshold':args.selection_args.max_std,
            'num trailing iterations':args.selection_args.evaluate_region,
            'epoch':None,
            'AUROC':None
        }
    }
    with open(model_tracking_fp, 'w') as fp:
        json.dump(tracking_info, fp, indent=2)
    # tracking_fp = os.path.join(("/").join(str(logger_args.save_dir).split("/")[:-1]), 'tracking.log')
    # # print(tracking_fp)
    # if os.path.exists(tracking_fp):
    #     with open(tracking_fp, 'r') as fp:
    #         tracking_info = json.load(fp)
    #     tracking_info["Models"][logger_args.experiment_name] = {
    #         "Training":{
    #             "Base_weights":args.model_args.ckpt_path,
    #             "max_epochs":args.optim_args.num_epochs,
    #             "random_state":args.random_state,
    #             "Started":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         }            
    #     }
    #     tracking_info['Last updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     with open(tracking_fp, 'w') as fp:
    #         json.dump(tracking_info, fp, indent=1)
    #     # print(json.dumps(tracking_info,indent=1))
    # else:
    #     tracking_info = None
    
    # Get logger.
    print ('Getting logger... log to path: {}'.format(logger_args.log_path))
    logger = Logger(logger_args.log_path, logger_args.save_dir)

    step_n =int(data_args.csv.replace(".csv","").split("_")[-1])
    if step_n == 0:
        # adjusted to allow the use of different pretraining
        if model_args.ckpt_path and model_args.ckpt_path == 'CheXpert':
            # specified CSL pretrained checkpoint
            print()
            print("pretrained checkpoint specified : {}".format(model_args.ckpt_path))
            # CL-specified args are used to load the model, rather than the
            # ones saved to args.json.
            model_args.pretrained = False
            ## CHANGE CheXpert checkpoint here!!
            model_args.ckpt_path = "/gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/checkpoint_0019.pth.tar"
            print(f"Loading model from {model_args.ckpt_path}")
            model, ckpt_info = ModelSaver.load_model(ckpt_path=model_args.ckpt_path,
                                                    gpu_ids=args.gpu_ids,
                                                    model_args=model_args,
                                                    is_training=True)
            # print(ckpt_info)
            optim_args.start_epoch = 1
        elif model_args.ckpt_path and model_args.ckpt_path == "MIMIC":
             # specified CSL pretrained checkpoint
            print('MIMIC not yet implemented')
            return
            print("pretrained checkpoint specified : {}".format(model_args.ckpt_path))
            # CL-specified args are used to load the model, rather than the
            # ones saved to args.json.
            model_args.pretrained = False
            ## CHANGE CSL checkpoint here!!
            ckpt_path = ""
            model, ckpt_info = ModelSaver.load_model(ckpt_path=model_args.ckpt_path,
                                                    gpu_ids=args.gpu_ids,
                                                    model_args=model_args,
                                                    is_training=True)
            optim_args.start_epoch = 1
        elif model_args.ckpt_path and model_args.ckpt_path == "ImageNet":
            # in the original MoCo CXR, they called this random initialization, but it used ImageNet pretraining
            print("Starting without pretrained training checkpoint, random initialization with ImageNet pretraining")
            model_fn = models.__dict__[model_args.model]
            if data_args.custom_tasks is not None:
                tasks = NamedTasks[data_args.custom_tasks]
            else:
                tasks = model_args.__dict__[TASKS]  # TASKS = "tasks"
            print("Tasks: {}".format(tasks))
            model = model_fn(tasks, model_args)
            #model = nn.DataParallel(model, args.gpu_ids)
            model = nn.DataParallel(model, args.gpu_ids).to(args.device)
        elif model_args.ckpt_path and model_args.ckpt_path == 'Random':
            # TODO
            print("WIP random initialization")
        else:
            print(f"Unrecognized ckpt_path: {args.model_args.ckpt_path}")

            
    else:  # # continual learning evaluation setup
        cur_step = "step_"+str(step_n)
        prev_step = "step_"+str(step_n-1)
        prev_step_mdl = os.path.join(str(logger_args.save_dir).replace(cur_step, prev_step), "best.pth.tar")
        model_args.ckpt_path = prev_step_mdl
        model_args.moco = False
        optim_args.start_epoch = 1
        print(f"Loading model from {model_args.ckpt_path}")
        model, ckpt_info = ModelSaver.load_model(ckpt_path=model_args.ckpt_path,
                                                gpu_ids=args.gpu_ids,
                                                model_args=model_args,
                                                is_training=True)
        print("ckpt_info:", ckpt_info)
        optim_args.start_epoch = 1

    # print(summary(model.module.cuda(), x=torch.rand(1,3, 320, 320).cuda()))
    # # For conaug, point to the MOCO pretrained weights.
    # if model_args.ckpt_path and model_args.ckpt_path != 'None':
       
    #     print("pretrained checkpoint specified : {}".format(model_args.ckpt_path))
    #     # CL-specified args are used to load the model, rather than the
    #     # ones saved to args.json.
    #     model_args.pretrained = False
    #     ckpt_path = model_args.ckpt_path
    #     model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
    #                                              gpu_ids=args.gpu_ids,
    #                                              model_args=model_args,
    #                                              is_training=True)
        

    #     if not model_args.moco:
    #         # optim_args.start_epoch = ckpt_info['epoch'] + 1
    #         # Adapted for continual learning
    #         optim_args.start_epoch = 1
    #     else:
    #         optim_args.start_epoch = 1
    # else:
    #     print('Starting without pretrained training checkpoint, random initialization.') # This is a random initialization of Imagenet Pretraining!
    #     # If no ckpt_path is provided, instantiate a new randomly
    #     # initialized model.
    #     model_fn = models.__dict__[model_args.model]
    #     if data_args.custom_tasks is not None:
    #         tasks = NamedTasks[data_args.custom_tasks]
    #     else:
    #         tasks = model_args.__dict__[TASKS]  # TASKS = "tasks"
    #     print("Tasks: {}".format(tasks))
    #     model = model_fn(tasks, model_args)
    #     #model = nn.DataParallel(model, args.gpu_ids)
    #     model = nn.DataParallel(model, args.gpu_ids).to(args.device)
    #     #model = nn.parallel.DistributedDataParallel(model, args.gpu_ids).to(args.device)


    # Put model on gpu or cpu and put into training mode.
    print(args.device)
    model = model.to(args.device)
    model.train()

    # print("========= MODEL ==========")
    # print(model)
    # print('==========================')
    
    # Get train and valid loader objects.
    train_loader = get_loader(phase="train",
                             data_args=data_args,
                             transform_args=transform_args,
                             is_training=True,
                             return_info_dict=True,
                             logger=logger)
    valid_loader = get_loader(phase="valid",
                              data_args=data_args,
                              transform_args=transform_args,
                              is_training=False,
                              return_info_dict=True,
                              logger=logger)
    
    # Instantiate the predictor class for obtaining model predictions.
    predictor = Predictor(model, args.device, args.code_dir)
    # Instantiate the evaluator class for evaluating models.
    evaluator = Evaluator(logger)
    # Get the set of tasks which will be used for saving models
    # and annealing learning rate.
    # allow for specific tasks to be designated as an argument
    if data_args.custom_tasks and data_args.custom_tasks not in NamedTasks:
        eval_tasks = data_args.custom_tasks.split(",")
    else:
        eval_tasks = EVAL_METRIC2TASKS[optim_args.metric_name]
    # Instantiate the saver class for saving model checkpoints.
    saver = ModelSaver(save_dir=logger_args.save_dir,
                       iters_per_save=logger_args.iters_per_save,
                       max_ckpts=logger_args.max_ckpts,
                       metric_name=optim_args.metric_name,
                       maximize_metric=optim_args.maximize_metric,
                       keep_topk=logger_args.keep_topk,
                       selection_args=args.selection_args)
    # get model layers to train
    if model_args.model == 'ResNet18':
        model_layers = [name for name,para in model.named_parameters()]
        #model_layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
        idxs = []
        for i, lyr in enumerate(model_layers):
            if lyr.endswith('bias'):
                if i == len(model_layers)-1:
                    continue
                if 'downsample' in model_layers[i+1]:
                    continue
                idxs.append(i+1)
        for ii, idx in enumerate(idxs):
            if ii == 0:
                layers = [','.join(model_layers[:idx])]          
            else:
                layers += [','.join(model_layers[idxs[ii-1]:idx])] 
        # the last fully-connected layer doesn't seem to be a part of ResNet, so it needs to be added
        layers += ["module.fc.weight,module.fc.bias"]
    # TODO: JBY: handle threshold for fine tuning
    if model_args.fine_tuning == 'full': # Fine tune all layers. 
        pass
    else:
        n_layers = int(model_args.fine_tuning.replace('last_',''))
        model_args.fine_tuning = ','.join(layers[-n_layers:])
        # Freeze other layers.
        models.PretrainedModel.set_require_grad_for_fine_tuning(model, model_args.fine_tuning.split(','))
    # print(summary(model.module.cuda(), x=torch.rand(1,3, 320, 320).cuda()))
    print('Total epochs of {}, Start LR {}, step size of {} with decay of {}'.format(optim_args.num_epochs, optim_args.lr, optim_args.lr_decay_step, optim_args.lr_decay_gamma))
    # Instantiate the optimizer class for guiding model training.
    optimizer = Optimizer(parameters=model.parameters(),
                          optim_args=optim_args,
                          batch_size=data_args.batch_size,
                          iters_per_print=logger_args.iters_per_print,
                          iters_per_visual=logger_args.iters_per_visual,
                          iters_per_eval=logger_args.iters_per_eval,
                          dataset_len=len(train_loader.dataset),
                          logger=logger)

    if model_args.ckpt_path and not model_args.moco and step_n == 0:
        # Load the same optimizer as used in the original training.
        optimizer.load_optimizer(ckpt_path=model_args.ckpt_path,
                                 gpu_ids=args.gpu_ids)

    model_uncertainty = model_args.model_uncertainty
    loss_fn = evaluator.get_loss_fn(loss_fn_name=optim_args.loss_fn,
                                    model_uncertainty=model_args.model_uncertainty,
                                    mask_uncertain=True,
                                    device=args.device)
    
    if step_n > 0:
        print('Measuring forward transfer capability and logging:')
        # # deploy to get the forward transfer assessment
        predictions, groundtruth = predictor.predict(valid_loader)
        metrics, curves = evaluator.evaluate_tasks(groundtruth, predictions)
        logger.log_metrics(metrics)

        # Add logger for all the metrics for valid_loader
        logger.log_scalars(metrics, optimizer.global_step, False)

        # Get the metric used to save model checkpoints.
        average_metric = evaluator.evaluate_average_metric(metrics,
                                                eval_tasks,
                                                optim_args.metric_name)
        print(f"AVG METRIC: {average_metric}")

    # Run training
    while not optimizer.is_finished_training():
        optimizer.start_epoch()

        # TODO: JBY, HACK WARNING  # What is the hack?
        metrics = None
        # for inputs, targets in train_loader:
        for inputs, targets, _ in train_loader:
            optimizer.start_iter()
            if optimizer.global_step and optimizer.global_step % optimizer.iters_per_eval == 0 or len(train_loader.dataset) - optimizer.iter < optimizer.batch_size:

                # Only evaluate every iters_per_eval examples.
                predictions, groundtruth, paths = predictor.predict(valid_loader, by_patient=args.by_patient)
                # print("predictions: {}".format(predictions))
                metrics, curves = evaluator.evaluate_tasks(groundtruth, predictions)
                # Log metrics to stdout.
                logger.log_metrics(metrics)

                # Add logger for all the metrics for valid_loader
                logger.log_scalars(metrics, optimizer.global_step, False)

                # Get the metric used to save model checkpoints.
                average_metric = evaluator.evaluate_average_metric(metrics,
                                                      eval_tasks,
                                                      optim_args.metric_name)
                # print(f"AVG METRIC: {average_metric}")

                if optimizer.global_step % logger_args.iters_per_save == 0:
                    # Only save every iters_per_save examples directly
                    # after evaluation.
                    print("Save global step: {}".format(optimizer.global_step))
                    saver.save(iteration=optimizer.global_step,
                               epoch=optimizer.epoch,
                               model=model,
                               optimizer=optimizer,
                               device=args.device,
                               metric_val=average_metric)
                    # make predictions on the entire validation set
                    import os
                    print("Predicting on entire validation set...")
                    predictions, gt, path = predictor.predict(valid_loader,by_patient=args.by_patient)
                    predictions.to_csv(os.path.join(saver.save_dir, "validation_predictions.csv"))
                    gt.to_csv(os.path.join(saver.save_dir, "valitaion_gt.csv"))

                # Step learning rate scheduler.
                optimizer.step_scheduler(average_metric)

            with torch.set_grad_enabled(True):
                logits, embedding = model(inputs.to(args.device))                
                loss = loss_fn(logits, targets.to(args.device))
                optimizer.log_iter(inputs, logits, targets, loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            optimizer.end_iter()

        optimizer.end_epoch(metrics)

        # if tracking_info is not None:
        #     tracking_info["Models"][logger_args.experiment_name]["Training"]["Progress"] = f"{optimizer.epoch}/{optimizer.num_epochs}"
        #     tracking_info['Last updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     with open(tracking_fp, 'w') as fp:
        #         json.dump(tracking_info, fp, indent=1)
        

    logger.log('=== Training Complete ===')
    if tracking_info is not None:
        tracking_info['training completed'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # TODO
        tracking_info['best iter info']['epoch'] = None
        tracking_info['best iter info']['AUROC'] = None
        with open(model_tracking_fp, 'w') as fp:
                json.dump(tracking_info, fp, indent=1)

if __name__ == '__main__':
    print("Beginning Training...")
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParser()
    train(parser.parse_args())
