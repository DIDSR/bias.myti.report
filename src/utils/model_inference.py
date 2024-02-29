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
import itertools
import timeit
# #
import torch.onnx
import onnx
import onnxruntime

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def inference_onnx(args):
    torch.cuda.set_device(args.gpu_id)
    # # # onnx checks
    # onnx_model = onnx.load(args.weight_file)
    # print(onnx.checker.check_model(onnx_model))

    # # Create dataset
    _dataset = Dataset(args.input_list_file, train_flag=False)
    # # Create data loader
    _loader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    print('Inferencing now ...')
    # # get ROI-aug and ROI-aug-avg
    auc_val, results_path = run_deploy_onnx(_loader, args)
    with open(os.path.join(os.path.dirname(args.log_path), 'inference_log.log'), 'a') as fp:
        fp.write("AUC\t{:1.5f}\t{}\n".format(auc_val, results_path))
    with open(args.log_path, 'a') as fp:
        fp.write(args.input_list_file + '\t' + args.weight_file +  '\t' +  results_path + '\t' + str(auc_val)  + '\n')


def run_deploy_onnx(data_loader, args):
    """ Function that deploys on the validation data loader, calculates sample based AUC and saves the scores in a tsv file.
    """
    start = timeit.default_timer()
    ort_session = onnxruntime.InferenceSession(args.weight_file,  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(' onnxruntime available providers: ' + str(ort_session.get_providers()))
    print(' onnxruntime running on: ' + onnxruntime.get_device())
    # #
    pid_all = []
    fnames_all = []
    type_all = []
    logits_all = []
    scores_all = []
    for i, (pid, fname, images, target) in enumerate(data_loader):
        # # compute ONNX Runtime output prediction
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images)}
        ort_outs = ort_session.run(None, ort_inputs)
        # # accumulate
        labl_list = list(target.cpu().numpy())
        type_all += labl_list
        pid_all += pid
        fnames_all += fname
        lgt = list(ort_outs[0])
        scr = torch.sigmoid(torch.flatten(torch.from_numpy(ort_outs[0])))
        lgt = list(itertools.chain(*lgt))
        logits_all += lgt        
        scr = list(scr.cpu().numpy())
        scores_all += scr
        
        
    
    # # calc AUC from ROC
    fpr, tpr, _ = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    print(' There were %d ROIs in the lists' % len(fnames_all))
    print(' AUROC = %f' % auc_val)

    # # save the score file
    result_df = pd.DataFrame(list(zip(pid_all, fnames_all, type_all, logits_all, scores_all)), columns=['patient_id', 'ROI_path', 'label', 'logits', 'score'])
    results_path = os.path.join(os.path.dirname(args.log_path), 'results__' + '.tsv')
    result_df.to_csv(results_path, sep='\t', index=False)
    # #
    stop = timeit.default_timer()
    print(' Time taken: ', stop - start) 
    return auc_val, results_path
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inference using onnx')
    parser.add_argument('-i', '--input_list_file', help='input list file', required=True)
    parser.add_argument('-w', '--weight_file', help='input weight file', required=True)
    parser.add_argument('-l', '--log_path', help='log saving path', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=48, help='batch size.')
    parser.add_argument('-t', '--threads', type=int, default=1, help='num. of threads.')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')

    args = parser.parse_args()
    print("Start inference...")

    # # save the args
    with open(os.path.join(os.path.dirname(args.log_path), 'myinference_args.json'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    # # ========================================
    inference_onnx(args)
    print('END.')
