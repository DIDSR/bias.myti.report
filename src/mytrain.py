'''
    Training program based on pytorch
    All params have brief descriptions in the argparse.ArgumentParser below
    Tensorboard can be used to track the training process
    
    Supported models:
            "googlenet" 
            "resnet18" 
            "wide_resnet50_2" 
            "densenet121" 
            "resnext50_32x4d"
    
    How are the pre-trained models adapted to a binary output:
        Implementation in add_classification_layer_v1 function: 
            - add the following layers in sequence: dropout (default p=0.2), linear(1000x512), linear(512x128), linear(128x1)
    
    Added functionality to do custom transfer learning, freezing at different points. Each DCNN structure requires
    custom code to work. The following are the supported models. This list will be updated as additional DCNNs are added.
            "resnet18"
            "densenet121"

    RKS, started Aug 1, 2022. 
    Git is used to track the versions.

    Worked in the following virtual environment:
        >> source /gpfs_projects/ravi.samala/venvs/venv_Pytorch/bin/activate
    
    When running the first time, pytorch will download the pretrained model

'''
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummaryX import summary
# #
from dat_data_load import Dataset
import os
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import json
# #
# # CONSTANTS
resnet18_ordered_layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
densenet121_ordered_layer_names = ['Conv2d_conv0', 'denseblock1', 'denseblock2', 'denseblock3', 'denseblock4', 'classifier']
master_iter = 0


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def apply_custom_transfer_learning__resnet18(net, custom_layer_name):
    first_non_frozen_layer_name = custom_layer_name[args.upto_freeze + 1]
    spacer = ''
    start_not_freezing_from_next_layer = False
    # # set the requires_grad to False to all first
    for param in net.parameters():
        param.requires_grad = False
    print('Partial fine tuning selected')
    print('Will freeze upto {} with the layer name of {}'.format(args.upto_freeze, first_non_frozen_layer_name))
    # # while iterating over the layers, check with the 
    for name, param in net.named_parameters():
        # # check if the names in custom_layer_name is in this name
        if len(custom_layer_name) > 0:
            for ii, each_layer_name in enumerate(custom_layer_name):
                # # the index in the name.split is important and may change
                # # based on the pretrained model
                if each_layer_name == name.split('.')[1]:
                    if each_layer_name == first_non_frozen_layer_name:
                        # # start un-freezing from here onwards
                        start_not_freezing_from_next_layer = True
                    # # found it
                    custom_layer_name.pop(ii)
                    spacer += '\t'
                    break
        if start_not_freezing_from_next_layer:
            param.requires_grad = True
            print('{} {} {}'.format(spacer, 'T', name))
        else:
            print('{} {} {}'.format(spacer, 'F', name))
    return net


def apply_custom_transfer_learning__densenet121(net, custom_layer_name):
    first_non_frozen_layer_name = custom_layer_name[args.upto_freeze + 1]
    spacer = ''
    start_not_freezing_from_next_layer = False
    # # set the requires_grad to False to all first
    for param in net.parameters():
        param.requires_grad = False
    print('Partial fine tuning selected')
    print('Will freeze upto {} with the layer name of {}'.format(args.upto_freeze, first_non_frozen_layer_name))
    # # while iterating over the layers, check with the 
    for name, param in net.named_parameters():
        # # check if the names in custom_layer_name is in this name
        if len(custom_layer_name) > 0:
            for ii, each_layer_name in enumerate(custom_layer_name):
                # # the index in the name.split is important and may change
                # # based on the pretrained model
                if each_layer_name == name.split('.')[2] or each_layer_name == name.split('.')[1]:
                    if each_layer_name == first_non_frozen_layer_name:
                        # # start un-freezing from here onwards
                        start_not_freezing_from_next_layer = True
                    # # found it
                    custom_layer_name.pop(ii)
                    spacer += '\t'
                    break
        if start_not_freezing_from_next_layer:
            param.requires_grad = True
            print('{} {} {}'.format(spacer, 'T', name))
        else:
            print('{} {} {}'.format(spacer, 'F', name))
    return net


def add_classification_layer_v1(model, num_channels, p=0.2):
    new_layers = nn.Sequential(nn.Dropout(p), nn.Linear(1000, 512), nn.Linear(512, 128), nn.Linear(128, num_channels))
    model = nn.Sequential(model, new_layers)
    return model


def train(args):
    # writer = SummaryWriter(log_dir=args.output_base_dir, flush_secs=1)
    # writer = SummaryWriter()
    # # based on the selected DNN N/W, modify the last layer of the ImageNet pre-trained DNN
    model = models.__dict__[args.dcnn](pretrained=True)
    num_channels = 1
    custom_layer_name = []
    if args.dcnn == 'googlenet':
        model = add_classification_layer_v1(model, num_channels)
    elif args.dcnn == 'resnet18':
        model = add_classification_layer_v1(model, num_channels)
        custom_layer_name = resnet18_ordered_layer_names.copy()
    elif args.dcnn == 'wide_resnet50_2':
        model = add_classification_layer_v1(model, num_channels)
    elif args.dcnn == 'densenet121':
        model = add_classification_layer_v1(model, num_channels)
        custom_layer_name = densenet121_ordered_layer_names.copy()
    elif args.dcnn == 'resnext50_32x4d':
        model = add_classification_layer_v1(model, num_channels)
    else:
        print('ERROR. UNKNOWN model.')
        return
    

    # # custom transfer learning >>
    if args.fine_tuning == 'partial':
        if args.dcnn == 'googlenet':
            print('ERROR. Custom transfer learning not implemented for this model.')
        elif args.dcnn == 'resnet18':
            model = apply_custom_transfer_learning__resnet18(model, custom_layer_name)
        elif args.dcnn == 'wide_resnet50_2':
            print('ERROR. Custom transfer learning not implemented for this model.')
        elif args.dcnn == 'densenet121':
            model = apply_custom_transfer_learning__densenet121(model, custom_layer_name)
        elif args.dcnn == 'resnext50_32x4d':
            print('ERROR. Custom transfer learning not implemented for this model.')
        else:
            print('ERROR. UNKNOWN model.')
            return
    elif args.fine_tuning == 'full':
        print('Full fine tuning selected')
    else:
        print('ERROR. UNKNOWN option for fine_tuning')
        return
    # # <<
    
    # # debug code to understand how a ROI passes through the network
    x=torch.rand(16,3,320,320)
    print(summary(model, x))
    # # 
    torch.cuda.set_device(args.gpu_id)
    model.cuda(args.gpu_id)
    # # Create tr and vd datasets
    train_dataset = Dataset(args.input_train_file, train_flag=True)
    valid_dataset = Dataset(args.validation_file, train_flag=False)
    # # Create tr and vd data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    num_steps_in_epoch = len(train_loader)
    # # select the optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, weight_decay=args.step_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.start_learning_rate, momentum=args.SGDmomentum)
    else:
        print('ERROR. UNKNOWN optimizer.')
        return
    # # learning rate scheduler
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_steps_in_epoch*args.decay_every_N_epoch, gamma=args.decay_multiplier)
    # #
    print('Training...')
    print('EPOCH\tTR-AVG-LOSS\tVD-AUC')
    criterion = nn.BCELoss()
    auc_val = -1
    for epoch in range(args.num_epochs):
        # # train for one epoch
        avg_loss = run_train(train_loader, model, criterion, optimizer, my_lr_scheduler)
        # # save
        if epoch % args.save_every_N_epochs == 0 or epoch == args.num_epochs-1:
            # # evaluate on validation set
            auc_val = run_validate(valid_loader, model, args)
            print("> {:d}\t{:1.5f}\t\t{:1.5f}".format(epoch, avg_loss, auc_val))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.dcnn,
                'state_dict': model.state_dict(),
                'auc': auc_val,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.output_base_dir, 'checkpoint__' + str(epoch) + '.pth.tar'))
    # # log the final model performance
    with open(args.log_path, 'a') as fp:
        fp.write(args.input_train_file + '\t' + args.validation_file +  '\t' +  args.output_base_dir + '\t' + str(auc_val) + '\n')


def run_train(train_loader, model, criterion, optimizer,  my_lr_scheduler):
    '''
        function that runs the training
    '''
    global master_iter

    # switch to train mode
    model.train()
    avg_loss = 0
    for i, (_, images, target) in enumerate(train_loader):
        # # measure data loading time
        master_iter += 1
        images = images.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(images.float())
        # # compute loss
        loss = criterion(torch.sigmoid(torch.flatten(output)), target.float())
        # writer.add_scalar("Loss/train", loss.item(), master_iter)
        avg_loss += loss.item()
        # # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        my_lr_scheduler.step()
        # #
        # writer.add_scalar("LR/train", my_lr_scheduler.get_last_lr()[0], master_iter)
    return avg_loss/len(train_loader)


def run_validate(val_loader, model, args):
    '''
        function the deploys on the input data loader
        calculates sample based AUC
        saves the scores in a tsv file
    '''
    global master_iter

    # # switch to evaluate mode
    model.eval()
    # #
    fnames_all = []
    type_all = []
    scores_all = []
    with torch.no_grad():
        for i, (fname, images, target) in enumerate(val_loader):
            # # compute output
            images = images.cuda()
            output = model(images.float())
            # #
            target_image_pred_probs = torch.sigmoid(torch.flatten(output))
            # # accumulate the scores
            labl_list = list(target.cpu().numpy())
            type_all += labl_list
            fnames_all += fname
            scr = list(target_image_pred_probs.cpu().numpy())
            scores_all += scr

    # # save the scores, labels in a tsv file
    if args.bsave_valid_results_at_epochs:
        result_df1 = pd.DataFrame(list(zip(fnames_all, type_all, scores_all)), columns=['ROI_path', 'type', 'probability'])
        results_path1 = os.path.join(args.output_base_dir, 'results__' + str(master_iter+1) + '.tsv')
        result_df1.to_csv(results_path1, sep='\t', index=False)
    # # calc AUC from ROC
    fpr, tpr, _ = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    with open(os.path.join(args.output_base_dir, 'log.log'), 'a') as fp:
        fp.write("{:d}\t{:1.5f}\n".format(master_iter, auc_val))
    # writer.add_scalar("AUC/test", auc_val, master_iter)
    return auc_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training using pytorch')
    parser.add_argument('-i', '--input_train_file', help='input training list file', required=True)
    parser.add_argument('-v', '--validation_file', help='input validation list file', required=True)
    parser.add_argument('-o', '--output_base_dir', help='output based dir', required=True)
    parser.add_argument('-d', '--dcnn', 
        help="which dcnn to use: 'googlenet', 'resnet18', 'wide_resnet50_2', 'resnext50_32x4d' or 'densenet121'", required=True)
    # parser.add_argument('-f', '--freeze_up_to', help="Must be a freezable layer in the structure e.g. FirstLayer", required=True)
    # Must be one of: 'FirstLayer', 'Mixed_3b', 'Mixed_3c', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'Mixed_5b', 'Mixed_5c'
    parser.add_argument('-f', '--fine_tuning', default='full', help="options: 'full' or 'partial'")
    parser.add_argument('-u', '--upto_freeze', type=int, default=0, 
        help="options: provide the layer number upto which to freeze")
    parser.add_argument('-l', '--log_path', help='log saving path', required=True)
    parser.add_argument('-p', '--optimizer', help='which optimizer to use: \'adam\' or \'sgd\'', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('-n', '--num_epochs', type=int, default=2000, help='num. of epochs.')
    parser.add_argument('-t', '--threads', type=int, default=4, help='num. of threads.')
    parser.add_argument('-r', '--start_learning_rate', type=float, default=0.0001, help='starting learning rate.')
    parser.add_argument('-s', '--step_decay', type=int, default=1000, help='Step for decay of learning rate.')
    parser.add_argument('--SGDmomentum', type=float, default=0.9, help='Momemtum param for SGD optimizer')
    parser.add_argument('--decay_every_N_epoch', type=int, default=5, help='Drop the learning rate every N epochs')
    parser.add_argument('--decay_multiplier', type=float, default=0.95, help='Decay multiplier')
    parser.add_argument('-e', '--save_every_N_epochs', type=int, default=1, help='save checkpoint every N number of epochs')
    parser.add_argument('--bsave_valid_results_at_epochs', type=bool, default=False, 
        help='save validation results csv at every epoch, True/False')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')

    args = parser.parse_args()
    print(args)

    # # save the args
    with open(os.path.join(args.output_base_dir, 'training_args.json'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    # # # ========================================
    train(args)
    print('END.')
