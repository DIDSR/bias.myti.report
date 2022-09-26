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



def train(args):
    import os # Why does os only import properly if in the function?
    """Run model training."""

    print("Start Training ...")

    # Get nested namespaces.
    model_args = args.model_args
    logger_args = args.logger_args
    optim_args = args.optim_args
    data_args = args.data_args
    transform_args = args.transform_args
    # # Changes made for continual_learning_evaluation =============================
    
    tracking_fp = os.path.join(("/").join(str(logger_args.save_dir).split("/")[:-1]), 'tracking.log')
    print(tracking_fp)
    if os.path.exists(tracking_fp):
        with open(tracking_fp, 'r') as fp:
            tracking_info = json.load(fp)
        tracking_info["Models"][logger_args.experiment_name] = {
            "Training":{
                "Started":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        tracking_info['Last updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(json.dumps(tracking_info,indent=1))
    else:
        tracking_info = None
    
    # # change json to csv if needed
    # # training
    # import os # TODO: figure out why os needs to be imported here for it to actually import?
    # tasks = model_args.__dict__[TASKS]
    # if not os.path.exists(data_args.csv):
    #     print(f"training csv {data_args.csv} does not exist!")
    #     if os.path.exists(data_args.csv.replace('.csv', '.json')):
    #         print("but a json version does! You probably need to run partitions_to_csv")
    #     return
    # Get logger.
    print ('Getting logger... log to path: {}'.format(logger_args.log_path))
    logger = Logger(logger_args.log_path, logger_args.save_dir)

    # For conaug, point to the MOCO pretrained weights.
    if model_args.ckpt_path and model_args.ckpt_path != 'None':
        # # continual learning evaluation setup
        # get step #
        step_n =int(data_args.csv.replace(".csv","").split("_")[-1])
        if step_n != 0:
            cur_step = "step_"+str(step_n)
            prev_step = "step_"+str(step_n-1)
            prev_step_mdl = os.path.join(str(logger_args.save_dir).replace(cur_step, prev_step), "best.pth.tar")
            model_args.ckpt_path = prev_step_mdl
            model_args.moco = False
        # #
        print("pretrained checkpoint specified : {}".format(model_args.ckpt_path))
        # CL-specified args are used to load the model, rather than the
        # ones saved to args.json.
        model_args.pretrained = False
        ckpt_path = model_args.ckpt_path
        model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                                 gpu_ids=args.gpu_ids,
                                                 model_args=model_args,
                                                 is_training=True)
        

        if not model_args.moco:
            # optim_args.start_epoch = ckpt_info['epoch'] + 1
            # Adapted for continual learning
            optim_args.start_epoch = 1
        else:
            optim_args.start_epoch = 1
    else:
        print('Starting without pretrained training checkpoint, random initialization.')
        # If no ckpt_path is provided, instantiate a new randomly
        # initialized model.
        model_fn = models.__dict__[model_args.model]
        if data_args.custom_tasks is not None:
            tasks = NamedTasks[data_args.custom_tasks]
        else:
            tasks = model_args.__dict__[TASKS]  # TASKS = "tasks"
        print("Tasks: {}".format(tasks))
        model = model_fn(tasks, model_args)
        #model = nn.DataParallel(model, args.gpu_ids)
        model = nn.DataParallel(model, args.gpu_ids).to(args.device)
        #model = nn.parallel.DistributedDataParallel(model, args.gpu_ids).to(args.device)


    # Put model on gpu or cpu and put into training mode.
    print(args.device)
    model = model.to(args.device)
    model.train()

    print("========= MODEL ==========")
    print(model)
    print('==========================')
    
    # Get train and valid loader objects.
    train_loader = get_loader(phase="train",
                             data_args=data_args,
                             transform_args=transform_args,
                             is_training=True,
                             return_info_dict=False,
                             logger=logger)
    valid_loader = get_loader(phase="valid",
                              data_args=data_args,
                              transform_args=transform_args,
                              is_training=False,
                              return_info_dict=False,
                              logger=logger)
    
    # Instantiate the predictor class for obtaining model predictions.
    predictor = Predictor(model, args.device, args.code_dir)
    # Instantiate the evaluator class for evaluating models.
    evaluator = Evaluator(logger)
    # Get the set of tasks which will be used for saving models
    # and annealing learning rate.
    eval_tasks = EVAL_METRIC2TASKS[optim_args.metric_name]
    # Instantiate the saver class for saving model checkpoints.
    saver = ModelSaver(save_dir=logger_args.save_dir,
                       iters_per_save=logger_args.iters_per_save,
                       max_ckpts=logger_args.max_ckpts,
                       metric_name=optim_args.metric_name,
                       maximize_metric=optim_args.maximize_metric,
                       keep_topk=logger_args.keep_topk)
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
    # Instantiate the optimizer class for guiding model training.
    optimizer = Optimizer(parameters=model.parameters(),
                          optim_args=optim_args,
                          batch_size=data_args.batch_size,
                          iters_per_print=logger_args.iters_per_print,
                          iters_per_visual=logger_args.iters_per_visual,
                          iters_per_eval=logger_args.iters_per_eval,
                          dataset_len=len(train_loader.dataset),
                          logger=logger)

    if model_args.ckpt_path and not model_args.moco:
        # Load the same optimizer as used in the original training.
        optimizer.load_optimizer(ckpt_path=model_args.ckpt_path,
                                 gpu_ids=args.gpu_ids)

    model_uncertainty = model_args.model_uncertainty
    loss_fn = evaluator.get_loss_fn(loss_fn_name=optim_args.loss_fn,
                                    model_uncertainty=model_args.model_uncertainty,
                                    mask_uncertain=True,
                                    device=args.device)
    
    # Run training
    while not optimizer.is_finished_training():
        optimizer.start_epoch()

        # TODO: JBY, HACK WARNING  # What is the hack?
        metrics = None
        for inputs, targets in train_loader:
            optimizer.start_iter()
            if optimizer.global_step and optimizer.global_step % optimizer.iters_per_eval == 0 or len(train_loader.dataset) - optimizer.iter < optimizer.batch_size:

                # Only evaluate every iters_per_eval examples.
                predictions, groundtruth = predictor.predict(valid_loader)
                # print("predictions: {}".format(predictions))
                metrics, curves = evaluator.evaluate_tasks(groundtruth, predictions)
                # Log metrics to stdout.
                logger.log_metrics(metrics)

                # Add logger for all the metrics for valid_loader
                logger.log_scalars(metrics, optimizer.global_step)

                # Get the metric used to save model checkpoints.
                average_metric = evaluator.evaluate_average_metric(metrics,
                                                      eval_tasks,
                                                      optim_args.metric_name)

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
                    predictions, gt = predictor.predict(valid_loader)
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
        if tracking_info is not None:
            tracking_info["Models"][logger_args.experiment_name]["Training"]["Progress"] = f"{optimizer.epoch}/{optimizer.num_epochs}"
            tracking_info['Last updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(tracking_fp, 'w') as fp:
                json.dump(tracking_info, fp, indent=1)
        

    logger.log('=== Training Complete ===')
    if tracking_info is not None:
        tracking_info["Models"][logger_args.experiment_name]["Training"]["Progress"] = "Complete " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tracking_info['Last updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(tracking_fp, 'w') as fp:
                json.dump(tracking_info, fp, indent=1)

if __name__ == '__main__':
    print("Beginning Training...")
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParser()
    train(parser.parse_args())
