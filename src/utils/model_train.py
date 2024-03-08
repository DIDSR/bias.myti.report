import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
#from collections import OrderedDict
# #
import os
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import json
# #
import torch.onnx
import onnx
import onnxruntime
import sys 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.utils.dat_data_load import Dataset
# # CONSTANTS
master_iter = 0

def strtobool(value: str) -> bool:
  """
  Change strings to boolean.
  
  Arguments
  =========
  value : str
  
  Returns
  =======
  bool
  """
  value = value.lower()
  if value in ("y", "yes", "on", "1", "true", "t"):
    return True
  return False
  

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the model checkpoint.
    
    Arguments
    =========
    state
        The checkpoint to save.
    filename : str
        The file name with which to the checkpoint.
    """
    torch.save(state, filename)


def modify_classification_layer_v1(model, num_channels):
    """
    Modify the last fully connected layer to have given output size.
    
    Arguments
    =========
    model
        Pytorch model to be modified.
    num_channels : int
        Number of outputs for the fully connected layer (the number of classes).

    Returns
    =======
    model
        Modified model.
    """
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_channels)
    return model
    
def apply_custom_transfer_learning__resnet18(net):
    """ 
    Set the ResNet18 model to freeze first certain number of layers.
    
    Arguments
    =========
    net
      Pytorch model to freeze.
    
    Returns
    =======
    net
    """
    # # get all the layer names
    model_layers = [name for name,para in net.named_parameters()]
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
    layers += ["fc.weight,fc.bias"]
    # # set requires_grad to False for freezing layers
    fine_tune_layers = ','.join(layers[args.freeze_up_to-len(layers):]).split(',')
    for name, param in net.named_parameters():
        if name not in fine_tune_layers:
            param.requires_grad = False
    # # make sure layers are correctly frozen
    parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
    assert len(parameters) == len(fine_tune_layers)
                
    return net


def load_custom_checkpoint(ckpt_path, base_dcnn, gpu_ids, num_channels):
    """ 
    Load customized pre-trained models with given weight file.
    
    Arguments
    =========
    ckpt_path : str
        File path of the pre-trained model.
    base_dcnn : str
        Name of the model architecture (e.g. resnet18, densenet121).
    gpu_ids : int
        Current GPU ID.
    num_channels : int
        Number of channels (classes).

    Returns
    =======
    model
        Loaded pre-trained model.
    
    """
    # # load model
    model = models.__dict__[base_dcnn](weights='IMAGENET1K_V1')
    model = modify_classification_layer_v1(model, num_channels)
    # # load state_dicts from the weight file
    device = f'cuda:{gpu_ids}'
    ckpt_dict = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt_dict['state_dict']    
    # # copy the weights and biases
    model.load_state_dict(state_dict, strict=False)
    return model
        

def train(args):
    """
    Main script for model training, including model selection, pre-trained weights loading,
    training/validation data loading, and model fine-tuning.
    
    Arguments
    =========
    args : argparse.Namespace
        The input arguments to the python script.
    """
    # set random state, if specified
    if args.random_state is not None:
        torch.manual_seed(args.random_state)
        torch.cuda.manual_seed_all(args.random_state)
        np.random.seed(args.random_state)
    # for result reproducibility    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # model selection
    num_channels = 1
    custom_layer_name = []
    if args.dcnn == 'googlenet':
        model = models.__dict__[args.dcnn](pretrained=args.pretrained_weights)
        model = modify_classification_layer_v1(model, num_channels=1)
    elif args.dcnn == 'resnet18':
        if args.pretrained_weights == True:
            model = models.__dict__[args.dcnn](weights='IMAGENET1K_V1')
        else:
            model = models.__dict__[args.dcnn]()
        model = modify_classification_layer_v1(model, num_channels=1)
    elif args.dcnn == 'wide_resnet50_2':
        model = models.__dict__[args.dcnn](pretrained=args.pretrained_weights)
        model = modify_classification_layer_v1(model, num_channels=1)
    elif args.dcnn == 'densenet121':
        model = models.__dict__[args.dcnn](pretrained=args.pretrained_weights)
        model = modify_classification_layer_v1(model, num_channels=1)
    elif args.dcnn == 'resnext50_32x4d':
        model = models.__dict__[args.dcnn](pretrained=args.pretrained_weights)
        model = modify_classification_layer_v1(model, num_channels=1)
    elif args.dcnn == 'CheXpert_Resnet':
        model = load_custom_checkpoint(args.custom_checkpoint_file, 'resnet18', args.gpu_id, num_channels=1)
        print('Using custom pretrained checkpoint file')
    else:
        print('ERROR. UNKNOWN model.')
        return    

    # # custom transfer learning >>
    if args.fine_tuning == 'partial':
        print(f"Fine tuning with first {args.freeze_up_to} layers frozen")
        if args.dcnn == 'resnet18' or args.dcnn == 'CheXpert_Resnet':
            model = apply_custom_transfer_learning__resnet18(model)
        else:
            print('ERROR. Custom transfer learning not implemented for this model.')
            return
    elif args.fine_tuning == 'full':
        print('Full fine tuning selected')
    else:
        print('ERROR. UNKNOWN option for fine_tuning')
        return
    
    torch.cuda.set_device(args.gpu_id)
    model.cuda(args.gpu_id)
    # # Create tr and vd datasets
    train_dataset = Dataset(args.input_train_file, train_flag=True, default_out_class=args.train_task)
    valid_dataset = Dataset(args.validation_file, train_flag=False, default_out_class=args.train_task)
    # # Create tr and vd data loaders   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    num_steps_in_epoch = len(train_loader)
    # # select the optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.start_learning_rate, momentum=args.SGDmomentum)
    else:
        print('ERROR. UNKNOWN optimizer.')
        return
    # # learning rate scheduler
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_every_N_epoch, gamma=args.decay_multiplier)
    # # start training
    print(f'Training for task: {args.train_task}')
    print('EPOCH\tTR-AVG-LOSS\tVD-AUC')
    criterion = nn.BCEWithLogitsLoss()
    auc_val = -1
    for epoch in range(args.num_epochs):
        # # train for one epoch
        avg_loss = run_train(train_loader, model, criterion, optimizer)
        my_lr_scheduler.step()
        # # validation and save checkpoint
        if epoch % args.save_every_N_epochs == 0 or epoch == args.num_epochs-1:
            # # evaluate on validation set
            auc_val = run_validate(valid_loader, model, args)
            print("> {:d}\t{:1.5f}\t\t{:1.5f}".format(epoch, avg_loss, auc_val))
            if epoch == args.num_epochs-1:
                checkpoint_file = os.path.join(args.output_base_dir, 'checkpoint__last.pth.tar')
            else:
                checkpoint_file = os.path.join(args.output_base_dir, 'checkpoint__' + str(epoch) + '.pth.tar')
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.dcnn,
                'state_dict': model.state_dict(),
                'auc': auc_val,
                'optimizer': optimizer.state_dict(),
            }, checkpoint_file)
            
    # # log the final model performance
    with open(args.log_path, 'a') as fp:
        fp.write(args.input_train_file + '\t' + args.validation_file +  '\t' +  args.output_base_dir + '\t' + str(auc_val) + '\n')
    
    # # save ONNX
    # Export the model
    onnx_model_path = os.path.join(args.output_base_dir, 'pytorch_last_epoch_model.onnx')
    x=torch.rand(16,3,320,320)
    torch.onnx.export(model,                   # model being run
                    x.cuda(),                  # model input (or a tuple for multiple inputs)
                    onnx_model_path,           # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'args.batch_size'},    # variable length axes
                                    'output' : {0 : 'args.batch_size'}})
    print('Final epoch model saved to: ' + onnx_model_path)


def run_train(train_loader, model, criterion, optimizer):
    """ 
    Function that runs the training
    
    Arguments
    =========
    train_loader : torch.utils.data.DataLoader
        Loaded training data set.
    model
        The model to be trained.
    criterion
        Loss function to be minimized.
    optimizer
        Training optimizer.

    Returns
    =======
    float
        Average loss for the current epoch.
    """
    global master_iter

    # switch to train mode
    model.train()
    avg_loss = 0
    for i, (_, _, images, target) in enumerate(train_loader):
        # # measure data loading time
        master_iter += 1
        images = images.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(images.float())
        # # compute loss
        loss = criterion(torch.flatten(output), target.float())
        avg_loss += loss.item()
        # # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        #my_lr_scheduler.step()
        # #
    return avg_loss/len(train_loader)


def run_validate(val_loader, model, args):
    """ 
    Deploys the model on the input data loader, calculates sample based AUC and saves the scores in a tsv file.
    
    Arguments
    =========
    val_loader : torch.utils.data.DataLoader
        Validation data.
    model
        The model to deploy.
    args : argparse.Namespace
        Input arguments to the python script.
    
    Returns
    =======
    float
        The calculated AUC value.
    """
    global master_iter

    # # switch to evaluate mode
    model.eval()
    # #
    pid_all = []
    fnames_all = []
    type_all = []
    logits_all = []
    scores_all = []
    with torch.no_grad():
        for i, (pid, fname, images, target) in enumerate(val_loader):
            # # compute output
            images = images.cuda()
            output = model(images.float())
            # #
            target_image_pred_logits = torch.flatten(output)
            target_image_pred_probs = torch.sigmoid(target_image_pred_logits)
            # # accumulate the scores
            labl_list = list(target.cpu().numpy())
            type_all += labl_list
            pid_all += pid
            fnames_all += fname
            logit = list(target_image_pred_logits.cpu().numpy())
            logits_all += logit
            scr = list(target_image_pred_probs.cpu().numpy())
            scores_all += scr

    # # save the scores, labels in a tsv file
    result_df1 = pd.DataFrame(list(zip(pid_all, fnames_all, type_all, logits_all, scores_all)), columns=['patient_id', 'ROI_path', 'label', 'logits', 'score'])
    if args.bsave_valid_results_at_epochs:        
        results_path1 = os.path.join(args.output_base_dir, 'results__' + str(master_iter+1) + '.tsv')
        result_df1.to_csv(results_path1, sep='\t', index=False)
    results_path2 = os.path.join(args.output_base_dir, 'results__last.tsv')
    result_df1.to_csv(results_path2, sep='\t', index=False)
    # # calc AUC from ROC
    fpr, tpr, _ = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    with open(os.path.join(args.output_base_dir, 'log.log'), 'a') as fp:
        fp.write("{:d}\t{:1.5f}\n".format(master_iter, auc_val))
    return auc_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training using pytorch')
    parser.add_argument('-i', '--input_train_file', help='input training list file', required=True)
    parser.add_argument('-v', '--validation_file', help='input validation list file', required=True)
    parser.add_argument('-o', '--output_base_dir', help='output based dir', required=True)
    parser.add_argument('-d', '--dcnn', default='CheXpert_Resnet',
        help="which dcnn to use: 'googlenet', 'resnet18', 'wide_resnet50_2', 'resnext50_32x4d', 'densenet121', 'CheXpert_Resnet'")        
    parser.add_argument('--freeze_up_to',  type=int, help="Specify the number of layers being frozen during training.")    
    parser.add_argument('--pretrained_weights', default=True, type=lambda x: bool(strtobool(x)), help="False if train from scratch.")
    parser.add_argument('-f', '--fine_tuning', default='full', help="options: 'full' or 'partial'")
    parser.add_argument('-l', '--log_path', help='log saving path', required=True)
    parser.add_argument('-p', '--optimizer', help='which optimizer to use: \'adam\' or \'sgd\'', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=48, help='batch size.')
    parser.add_argument('-n', '--num_epochs', type=int, default=10, help='num. of epochs.')
    parser.add_argument('-t', '--threads', type=int, default=1, help='num. of threads.')
    parser.add_argument('-r', '--start_learning_rate', type=float, default=5e-5, help='starting learning rate.')
    parser.add_argument('-s', '--step_decay', type=int, default=3, help='Step for decay of learning rate.')
    parser.add_argument('--SGDmomentum', type=float, default=0.9, help='Momemtum param for SGD optimizer')
    parser.add_argument('--decay_every_N_epoch', type=int, default=3, help='Drop the learning rate every N epochs')
    parser.add_argument('--decay_multiplier', type=float, default=0.2, help='Decay multiplier')
    parser.add_argument('-e', '--save_every_N_epochs', type=int, default=1, help='save checkpoint every N number of epochs')
    parser.add_argument('--bsave_valid_results_at_epochs', type=lambda x: bool(strtobool(x)), default=False, 
        help='save validation results csv at every epoch, True/False')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-c', '--custom_checkpoint_file',  
        help='custom checkpoint file to start')
    parser.add_argument('--random_state', type=int, default=None)
    parser.add_argument('--train_task', type=str, default='Yes', help='specify training task')
    

    args = parser.parse_args()
    print("Start experiment...")
    
    # # create the output dirctory if not exist
    if not os.path.exists(args.output_base_dir):
        os.makedirs(args.output_base_dir)

    # # save the args
    with open(os.path.join(args.output_base_dir, 'training_args.json'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    # # # ========================================
    train(args)
    print('END.')
