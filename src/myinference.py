'''
    Program that deploys breast mass ROIs from mammography and DBT to a trained model
'''
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
# #
from dat_data_load import Dataset
import os
import argparse
import json
import numpy as np
import pandas as pd
from sklearn import metrics


def add_classification_layer_v1(model, num_channels, p=0.2):
    new_layers = nn.Sequential(nn.Dropout(p), nn.Linear(1000, 512), nn.Linear(512, 128), nn.Linear(128, num_channels))
    model = nn.Sequential(model, new_layers)
    return model


def inference(args):
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
    elif args.dcnn == 'resnext50_32x4d':
        model = add_classification_layer_v1(model, num_channels)
    else:
        print('ERROR. UNKNOWN model.')
        return
    
    # # 
    torch.cuda.set_device(args.gpu_id)
    checkpoint = torch.load(args.weight_file)
    model.load_state_dict(checkpoint['state_dict'])
    # model.load_state_dict(torch.load(args.weight_file))
    model.cuda(args.gpu_id)
    # # Create dataset
    # if args.mode:
    #     _dataset = Dataset(args.input_list_file, crop_to_224=True, train_flag=True, custom_scale=True)
    # else:
    #     _dataset = Dataset(args.input_list_file, crop_to_224=True, train_flag=False, custom_scale=True)
    # # Create data loaders
    tr_dataset = Dataset(args.input_list_file, crop_to_224=True, train_flag=True, custom_scale=True)
    vd_dataset = Dataset(args.input_list_file, crop_to_224=True, train_flag=False, custom_scale=True)
    tr_data_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    vd_data_loader = DataLoader(vd_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)

    print('Inference...')
    auc_val, auc_val2, results_path = run_deploy(tr_data_loader, model, args, True)
    auc_val_center, auc_val2_center, _ = run_deploy(vd_data_loader, model, args, False)
    print("{} {:1.5f} {:1.5f} {:1.5f}".format(results_path, auc_val_center, auc_val, auc_val2))
    # # log the final model
    with open(args.log_path, 'a') as fp:
        fp.write(args.input_list_file + '\t' + args.weight_file +  '\t' +  results_path + '\t' + str(auc_val_center) + '\t' + str(auc_val)  + '\t' + str(auc_val2) + '\n')


def run_deploy(data_loader, model, args, mode):

    if mode:
        num_augs = 20
    else:
        num_augs = 1
    # # switch to evaluate mode
    model.eval()
    # #
    fnames_all = []
    type_all = []
    scores_all = []
    with torch.no_grad():
        for augs in range(num_augs):
            for i, (fname, images, target) in enumerate(data_loader):
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

    chk_pt_name = os.path.basename(args.weight_file)
    output_dir = os.path.dirname(args.weight_file)
    results_path1 = os.path.join(output_dir, 'results_ROI__' + chk_pt_name + '.tsv')
    if mode:
        result_df1 = pd.DataFrame(list(zip(fnames_all, type_all, scores_all)), columns=['ROI_path', 'type', 'probability'])
        result_df1.to_csv(results_path1, sep='\t', index=False)
    # # ROC by ROI
    print('There are %d ROIs in the lists' % len(fnames_all))
    fpr, tpr, _ = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    # #
    # # check for augmented samples
    unq_roi_names = []
    for each_view in fnames_all:
        if each_view.split('__')[0] not in unq_roi_names:
            unq_roi_names.extend([each_view.split('__')[0]])
    print('There are %d unique ROIs without augmentation in the list' % len(unq_roi_names))
    # # average the augmented sample scores
    scores_avg = np.zeros((len(unq_roi_names), ), dtype=np.float32)
    labels_avg = np.zeros((len(unq_roi_names),), dtype=np.float32)
    count = np.zeros((len(unq_roi_names), ), dtype=np.float32)
    for countr, each_roi in enumerate(fnames_all):
        idx = unq_roi_names.index(each_roi.split('__')[0])
        scores_avg[idx] += scores_all[countr]
        labels_avg[idx] += type_all[countr]
        count[idx] += 1.0
    scores_avg = scores_avg / count
    labels_avg = labels_avg / count
    auc_val2 = metrics.roc_auc_score(labels_avg, scores_avg)
    dat = np.column_stack((unq_roi_names, scores_avg, labels_avg))
    results_path2 = os.path.join(output_dir, 'results_ROIaug__' + chk_pt_name + '.tsv')
    if mode:
        np.savetxt(results_path2, dat, delimiter="\t", fmt='%s')

    # # #
    # case_names = []
    # for each_view in unq_roi_names:
    #     if each_view.split('_')[0][0:6] not in case_names:
    #         case_names.extend([each_view.split('_')[0][0:6]])
    # print('There are %d views in the list' % len(case_names))
    # scores_avg2 = np.zeros((len(case_names), ), dtype=np.float32)
    # labels_avg2 = np.zeros((len(case_names),), dtype=np.float32)
    # count2 = np.zeros((len(case_names), ), dtype=np.float32)
    # for countr, each_roi in enumerate(unq_roi_names):
    #     idx = case_names.index(each_roi.split('_')[0][0:6])
    #     scores_avg2[idx] += scores_avg[countr]
    #     if labels_avg[countr] == 1.0:
    #         labels_avg2[idx] = 1
    #     else:
    #         labels_avg2[idx] = 0
    #     count2[idx] += 1.0
    # scores_avg2 = scores_avg2 / count2
    # auc_val2 = metrics.roc_auc_score(labels_avg, scores_avg)
    # auc_val3 = metrics.roc_auc_score(labels_avg2, scores_avg2)
    # dat = np.column_stack((case_names, scores_avg2, labels_avg2))
    # results_path3 = os.path.join(output_dir, 'results_byview__' + chk_pt_name + '.tsv')
    # np.savetxt(results_path3, dat, delimiter="\t", fmt='%s')
    print('AUC = %f (using roc_auc_score): roi-based' % auc_val)
    print('AUC = %f (using roc_auc_score): roi-based (x rot -avg)' % auc_val2)
    # print('AUC = %f (using roc_auc_score): view-based (x rot -avg)' % auc_val3)

    # #
    return auc_val, auc_val2, results_path2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference using tf')
    parser.add_argument('-i', '--input_list_file', help='input list file', required=True)
    parser.add_argument('-w', '--weight_file', help='input weight file', required=True)
    parser.add_argument('-d', '--dcnn', help="which dcnn to use: 'googlenet', 'resnet18', 'wide_resnet50_2' or 'densenet121'", required=True)
    parser.add_argument('-l', '--log_path', help='log saving path', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('-t', '--threads', type=int, default=4, help='num. of threads.')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-m', '--mode', type=bool, default=True, help='For training use True, for validation use False')

    args = parser.parse_args()
    print(args)

    # # # save the args
    # with open(os.path.join(args.output_base_dir, 'inferences_args.json'), 'w') as fp:
    #     json.dump(args.__dict__, fp, indent=2)
    # # # ========================================
    inference(args)
    print('END.')
