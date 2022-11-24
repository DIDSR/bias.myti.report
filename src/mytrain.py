import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.ops as ops
from torch.utils.data import DataLoader
from torchmetrics import AUROC
# import webdataset as wds
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
import shutil
# #

master_iter = 0


def change_googlenet_model(model, num_channels):
    # # add a flatten and linear layer at the end
    # model = nn.Sequential(model, nn.Flatten(1, -1), nn.Linear(1000, num_channels))
    # model = nn.Sequential(model, nn.Linear(1000, num_channels))

    features = nn.ModuleList(model.children())[:-1]
    model_features = nn.Sequential(*features) 
    # new_layers = nn.Sequential(nn.Linear(1000, num_channels)) 
    # model = nn.Sequential(model_features,  nn.Flatten(1, -1), nn.Linear(1000, num_channels))
    # model = nn.Sequential(model_features,  nn.Flatten(0, -1), nn.Linear(1024, 512), nn.Linear(512, num_channels))
    new_layers = nn.Sequential(nn.Flatten(1, -1), nn.Linear(1024, 512), nn.Linear(512, num_channels)) 
    # model = nn.Sequential(model_features,  nn.Flatten(1, -1), nn.Linear(1024, 512), nn.Linear(512, num_channels))
    model = nn.Sequential(model_features, new_layers)
    return model


def change_resnet18_model(model, num_channels, p=0.2):
    new_layers = nn.Sequential(nn.Dropout(p), nn.Linear(1000, 512), nn.Linear(512, 128), nn.Linear(128, num_channels))
    model = nn.Sequential(model, new_layers)
    return model

# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.start_learning_rate * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def roc_auc(output, target, topk=(1,)):
    with torch.no_grad():
        pred_np = list(output.cpu().numpy())
        pred_np = [list(x)[1] for x in pred_np]
        target_np = list(target.cpu().numpy())
        fpr, tpr, thresholds = metrics.roc_curve(np.array(target_np), np.array(pred_np), pos_label=1)
        res = []
        AUC = metrics.auc(fpr, tpr)
        res.append(AUC)
    return res


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train(args):
    writer = SummaryWriter(log_dir=args.output_base_dir, flush_secs=1)
    # # based on the selected DNN N/W, modify the last layer of the ImageNet pre-trained DNN
    model = models.__dict__[args.dcnn](pretrained=True)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    # for p in model.parameters():
    #     p.requires_grad = False
    num_channels = 1
    if args.dcnn == 'googlenet':
        model = change_googlenet_model(model, num_channels)
    elif args.dcnn == 'resnet18':
        model = change_resnet18_model(model, num_channels)
    
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
    # optimizer = torch.optim.Adam(model.parameters(), args.start_learning_rate, weight_decay=args.step_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.start_learning_rate, momentum=0.9)
    decayRate = 0.95
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_steps_in_epoch*5, gamma=decayRate)

    print('Training...')
    # criterion = nn.CrossEntropyLoss()
    # loss = ops.sigmoid_focal_loss(inputs=output.float(), targets=nn.functional.one_hot(target, num_classes=2).float(),
    #                               alpha=args.alpha, gamma=args.gamma, reduction='mean')
    criterion = nn.BCELoss()
    for epoch in range(args.num_epochs):
        # adjust_learning_rate(optimizer, epoch, args)
        # # train for one epoch
        run_train(train_loader, model, criterion, optimizer, epoch, writer, my_lr_scheduler, num_steps_in_epoch, valid_loader, args)
        # break
    
        # # save
        if epoch % args.save_every_N_epochs == 0 or epoch == args.num_epochs-1:
            # # evaluate on validation set
            auc_val = run_validate(valid_loader, model, writer, args)
            print("> {:d}\t{:1.5f}".format(epoch, auc_val))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.dcnn,
                'state_dict': model.state_dict(),
                'auc': auc_val,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.output_base_dir, 'checkpoint__' + str(epoch) + '.pth.tar'))


def run_train(train_loader, model, criterion, optimizer, epoch, writer, my_lr_scheduler, num_steps_in_epoch, val_loader, args):
    global master_iter

    # switch to train mode
    model.train()
    for i, (_, images, target) in enumerate(train_loader):
        # # measure data loading time
        master_iter += 1
        images = images.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output = model(images.float())
        
        loss = criterion(torch.sigmoid(torch.flatten(output)), target.float())
        writer.add_scalar("Loss/train", loss.item(), master_iter)
        # # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        my_lr_scheduler.step()
        # #
        writer.add_scalar("LR/train", my_lr_scheduler.get_last_lr()[0], master_iter)


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

    result_df1 = pd.DataFrame(list(zip(fnames_all, type_all, scores_all)), columns=['ROI_path', 'type', 'probability'])
    results_path1 = os.path.join(args.output_base_dir, 'results__' + str(master_iter+1) + '.tsv')
    result_df1.to_csv(results_path1, sep='\t', index=False)
    # #
    fpr, tpr, thresholds = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    with open(os.path.join(args.log_path, 'log.log'), 'a') as fp:
        fp.write("{:d}\t{:1.5f}".format(master_iter, auc_val))
    writer.add_scalar("AUC/test", auc_val, master_iter)
    return auc_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference using tf')
    parser.add_argument('-i', '--input_train_file', help='input training list file', required=True)
    parser.add_argument('-v', '--validation_file', help='input validation list file', required=True)
    parser.add_argument('-o', '--output_base_dir', help='output dir', required=True)
    parser.add_argument('-d', '--dcnn', help="which dcnn to use: 'inception_v1', 'inception_v4', 'alexnet' or 'inception_resnet_v2'", required=True)
    # parser.add_argument('-f', '--freeze_up_to', help="Must be a freezable layer in the structure e.g. FirstLayer", required=True)
    # Must be one of: 'FirstLayer', 'Mixed_3b', 'Mixed_3c', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'Mixed_5b', 'Mixed_5c'
    # parser.add_argument('-g', '--ckpt_path', help='checkpoint saving path', required=True)
    parser.add_argument('-l', '--log_path', help='log saving path', required=True)
    # parser.add_argument('-o', '--optimizer', help='which optimizer to use: \'adam\' or \'gd\'', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('-n', '--num_epochs', type=int, default=2000, help='num. of epochs.')
    parser.add_argument('-t', '--threads', type=int, default=4, help='num. of threads.')
    parser.add_argument('-r', '--start_learning_rate', type=float, default=0.0001, help='starting learning rate.')
    parser.add_argument('-s', '--step_decay', type=int, default=1000, help='Step for decay of learning rate.')
    # parser.add_argument('--alpha', type=float, default=0.25, help='focal loss alpha')
    # parser.add_argument('--gamma', type=float, default=7.0, help='focal loss gamma')
    parser.add_argument('-e', '--save_every_N_epochs', type=int, default=1, help='save checkpoint every N number of epochs')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')

    # example command: python Finetune_inception4_CDR_v1.py -i /nas/unas21/rsamala/tf/RKS/UND_exp2/tfrecords2/SFM_USF_DM_MB_v4_not909_v2_256_dp.tfrecords -d inception_v1 -l /nas/unas21/calebric/temp/inception_v1_finetuned_w_images -o adam
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)

    with open(os.path.join(args.log_path, 'args.json'), 'a') as fp:
        json.dump(args.__dict__, fp, indent=2)
    # # # ========================================
    # # # redirect the logging for those long runs
    # # # ========================================
    # # get TF logger
    # log = logging.getLogger('tensorflow')
    # log.setLevel(logging.DEBUG)
    # # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # # create file handler which logs even debug messages
    # fh = logging.FileHandler(os.path.join(args.log_path, 'tensorflow.log'), mode='a')
    # fh.setLevel(logging.DEBUG)
    # fh.setFormatter(formatter)
    # log.addHandler(fh)
    # # # ========================================
    train(args)
    print('END.')