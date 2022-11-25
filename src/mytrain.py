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

# # Supported models:
# # AlexNet (alexnet)
# # ConvNeXt
# # DenseNet
# # EfficientNet
# # EfficientNetV2
# # GoogLeNet
# # Inception V3
# # MaxVit
# # MNASNet
# # MobileNet V2
# # MobileNet V3
# # RegNet
# # ResNet
# # ResNeXt
# # ShuffleNet V2
# # SqueezeNet
# # SwinTransformer
# # VGG
# # VisionTransformer
# # Wide ResNet

master_iter = 0


def add_classification_layer_v1(model, num_channels, p=0.2):
    new_layers = nn.Sequential(nn.Dropout(p), nn.Linear(1000, 512), nn.Linear(512, 128), nn.Linear(128, num_channels))
    model = nn.Sequential(model, new_layers)
    return model


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(args):
    writer = SummaryWriter(log_dir=args.output_base_dir, flush_secs=1)
    # # based on the selected DNN N/W, modify the last layer of the ImageNet pre-trained DNN
    model = models.__dict__[args.dcnn](pretrained=True)
    num_channels = 1
    if args.dcnn == 'googlenet':
        model = add_classification_layer_v1(model, num_channels)
    elif args.dcnn == 'resnet18':
        model = add_classification_layer_v1(model, num_channels)
    elif args.dcnn == 'wide_resnet50_2':
        model = add_classification_layer_v1(model, num_channels)
    elif args.dcnn == 'densenet121':
        model = add_classification_layer_v1(model, num_channels)
    else:
        print('ERROR. UNKNOWN model.')
        return
    
    # # # debug code to understand how a ROI passes through the network
    x=torch.rand(16,3,224,224)
    print(summary(model, x))
    # # 
    torch.cuda.set_device(args.gpu_id)
    model.cuda(args.gpu_id)
    # # Create tr and vd datasets
    train_dataset = Dataset(args.input_train_file, crop_to_224=True, train_flag=True, custom_scale=True)
    valid_dataset = Dataset(args.validation_file, crop_to_224=True, train_flag=False, custom_scale=True)
    # # Create tr and vd data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    num_steps_in_epoch = len(train_loader)
    # # 
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, weight_decay=args.step_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.start_learning_rate, momentum=args.SGDmomentum)
    else:
        print('ERROR. Uknown optimizer')
        return
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_steps_in_epoch*args.decay_every_N_epoch, gamma=args.decay_multiplier)
    # #
    print('Training...')
    print('EPOCH\tTR-AVG-LOSS\tVD-AUC')
    criterion = nn.BCELoss()
    auc_val = -1
    for epoch in range(args.num_epochs):
        # # train for one epoch
        avg_loss = run_train(train_loader, model, criterion, optimizer, epoch, writer, my_lr_scheduler, num_steps_in_epoch, valid_loader, args)
        # # save
        if epoch % args.save_every_N_epochs == 0 or epoch == args.num_epochs-1:
            # # evaluate on validation set
            auc_val = run_validate(valid_loader, model, writer, args)
            print("> {:d}\t{:1.5f}\t\t{:1.5f}".format(epoch, avg_loss, auc_val))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.dcnn,
                'state_dict': model.state_dict(),
                'auc': auc_val,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.output_base_dir, 'checkpoint__' + str(epoch) + '.pth.tar'))
    # # log the final model
    with open(args.log_path, 'a') as fp:
        fp.write(args.input_train_file + '\t' + args.validation_file +  '\t' +  args.output_base_dir + '\t' + str(auc_val) + '\n')


def run_train(train_loader, model, criterion, optimizer, epoch, writer, my_lr_scheduler, num_steps_in_epoch, val_loader, args):
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
        writer.add_scalar("Loss/train", loss.item(), master_iter)
        avg_loss += loss.item()
        # # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        my_lr_scheduler.step()
        # #
        writer.add_scalar("LR/train", my_lr_scheduler.get_last_lr()[0], master_iter)
    return avg_loss/len(train_loader)


def run_validate(val_loader, model, writer, args):
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
            # # accumulate
            labl_list = list(target.cpu().numpy())
            type_all += labl_list
            fnames_all += fname
            scr = list(target_image_pred_probs.cpu().numpy())
            scores_all += scr

    if args.bsave_valid_results_at_epochs:
        result_df1 = pd.DataFrame(list(zip(fnames_all, type_all, scores_all)), columns=['ROI_path', 'type', 'probability'])
        results_path1 = os.path.join(args.output_base_dir, 'results__' + str(master_iter+1) + '.tsv')
        result_df1.to_csv(results_path1, sep='\t', index=False)
    # #
    fpr, tpr, _ = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    with open(os.path.join(args.output_base_dir, 'log.log'), 'a') as fp:
        fp.write("{:d}\t{:1.5f}\n".format(master_iter, auc_val))
    writer.add_scalar("AUC/test", auc_val, master_iter)
    return auc_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training using pytorch')
    parser.add_argument('-i', '--input_train_file', help='input training list file', required=True)
    parser.add_argument('-v', '--validation_file', help='input validation list file', required=True)
    parser.add_argument('-o', '--output_base_dir', help='output dir', required=True)
    parser.add_argument('-d', '--dcnn', help="which dcnn to use: 'googlenet', 'resnet18', 'wide_resnet50_2' or 'densenet121'", required=True)
    # parser.add_argument('-f', '--freeze_up_to', help="Must be a freezable layer in the structure e.g. FirstLayer", required=True)
    # Must be one of: 'FirstLayer', 'Mixed_3b', 'Mixed_3c', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'Mixed_5b', 'Mixed_5c'
    # parser.add_argument('-g', '--ckpt_path', help='checkpoint saving path', required=True)
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
    parser.add_argument('--bsave_valid_results_at_epochs', type=bool, default=False, help='save validation results csv at every epoch, True/False')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')

    args = parser.parse_args()
    print(args)

    # # save the args
    with open(os.path.join(args.output_base_dir, 'training_args.json'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    # # # ========================================
    train(args)
    print('END.')
