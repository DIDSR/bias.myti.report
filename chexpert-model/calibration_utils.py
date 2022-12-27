import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from torchmetrics.functional import calibration_error
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle

def calibration_metrics(predictions, labels, subgroups=None):
    '''Returns AUROC, Brier, ECE, and NLL scores for binary classification tasks'''
    if len(np.unique(labels)) == 1:
        auroc = np.nan
    else:
        auroc = roc_auc_score(labels, predictions)      
    brier = brier_score_loss(labels, predictions, pos_label=1)
    # convert to tensors
    p = torch.tensor(predictions)
    l = torch.tensor(labels)
    ece = calibration_error(target=l.int(), preds=p, norm='l1').item()
    nll_criterion = nn.BCELoss()
    nll = nll_criterion(p, l.double()).item()
    out = pd.DataFrame({"AUROC":[auroc], 'Brier':[brier], 'ECE':[ece], 'NLL':[nll]})
    return out

class PlattScaling():
    def __init__(self, random_state=0):
        self.model = LogisticRegression(random_state=random_state)

    def fit(self, predictions, labels):
        self.model = self.model.fit(predictions, labels)

    def scale(self, predictions):
        return self.model.predict_proba(predictions)

    def plot_curves(self, info_dict, filepath):
        '''info_dict should be in the format: 
            {'part_label (ex. validation)':{'labels':label_array, 'preds':raw predictions}}'''
        fig, ax = plt.subplots(figsize=(6,8))
        # add reference line
        line = mlines.Line2D([0,1], [0,1], color='black', linestyle='--', linewidth=0.5)
        ax.add_line(line)   
        for ii, part in enumerate(info_dict):
            info = info_dict[part]
            info['probs'] = self.scale(info['preds'])
            y_prob_true, x_prob_pred = calibration_curve(info['labels'], info['probs'][:,1])
            ax.plot(x_prob_pred, y_prob_true, marker='o',linewidth=1, label=part)
        plt.title("Calibration Curve (Platt Scaling)")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("True Probability in each bin")
        plt.legend()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    def save(self, save_fp):
        with open(save_fp, 'wb') as fp:
            pickle.dump(self.model, fp)
    
    def load(self, file):
        with open(file, 'rb') as fp:
            self.model = pickle.load(fp)

class CalibratedModel(nn.Module):
    def __init__(self, model=None, calibration_mode='temperature'):
        super(CalibratedModel, self).__init__()
        self.model = model
        self.calibration_mode = calibration_mode
        if self.calibration_mode == 'temperature':
            self.temperature = nn.Parameter(torch.ones(1)*1.5)

    def forward(self, input):
        if self.calibration_mode == 'temperature':
            logits = self.logits(input)
            return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def ece(self, y_true, y_pred): 
        return calibration_error(target=y_true.int(), preds=y_pred, norm='l1')
    
    def brier(self, y_true, y_prob, pos_label=1):
        return brier_score_loss(y_true.cpu().int().numpy(), y_prob.cpu().detach().numpy(), pos_label=pos_label)

    def get_metrics(self, logits, labels):
        metrics = {}
        metrics['ECE'] = self.ece(labels, logits).detach().cpu().numpy().item()
        metrics['Brier'] = self.brier(labels, logits, pos_label=1)
        nll_criterion = nn.CrossEntropyLoss().cuda()
        if len([logits.shape]) == 1:
            # binary classification
            nll_criterion = nn.BCELoss().cuda()
        metrics['NLL'] = nll_criterion(logits, labels).item()
        return metrics

    def set_temperature(self, valid_loader=None, valid_results=None, lr=0.01, max_iter=50):
        self.temp_metrics = {}
        # NOTE: Brier score only works for binary classificaiton, assumes first column is positive percentage
        self.cuda()
        self.temp_lr = lr
        self.temp_max_iter = max_iter
        nll_criterion = nn.CrossEntropyLoss().cuda()
        if valid_results is None:
            logits_list = []
            labels_list = []
            with torch.no_grad():
                for input, label in valid_loader:
                    input = input.cuda()
                    logits = self.model(input)
                    logits_list.append(logits)
                    labels_list.append(label)
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda().argmax(dim=1).long()
        else: # expects valid_predictions = [preds, labels]
            if len([valid_results[1].shape]) != 1:
                labels = torch.argmax(valid_results[1], dim=1).cuda()
            else:
                labels = valid_results[1].cuda()
            logits = valid_results[0].cuda()
        if len([logits.shape]) == 1:
            # binary classification
            nll_criterion = nn.BCELoss().cuda()
        # calculate before-scaling metrics
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = self.ece(labels, logits)
        before_temperature_brier = self.brier(labels, logits[:,0], pos_label=1)# needs to be changed depending on exact model task set up
        self.temp_metrics['before'] = {
            'ECE':before_temperature_ece.detach().cpu().numpy().item(),
            'Brier':before_temperature_brier, 
            'NLL':before_temperature_nll
        }

        print(f"\nBefore temperature scaling:\nNLL: {before_temperature_nll:.4f}\nECE: {before_temperature_ece:.4f}\nBrier: {before_temperature_brier:.4f}")
        optimizer = optim.LBFGS([self.temperature], lr=self.temp_lr, max_iter=self.temp_max_iter)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        # calculate after-scaling metrics
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        ch_nll = after_temperature_nll - before_temperature_nll
        after_temperature_ece = self.ece(labels, self.temperature_scale(logits))
        ch_ece = after_temperature_ece - before_temperature_ece
        after_temperature_brier = self.brier(labels, self.temperature_scale(logits)[:,0], pos_label=1)
        ch_brier = after_temperature_brier - before_temperature_brier
        print(f"\nAfter temperature scaling:\nNLL: {after_temperature_nll:.4f} ({ch_nll})\nECE: {after_temperature_ece:.4f} ({ch_ece})\nBrier: {after_temperature_brier:.4f} ({ch_brier})")
        print(f"Temperature: {self.temperature.item():.4f}")
        print(f"(max_iter={self.temp_max_iter}, lr={self.temp_lr})\n")
        self.temp_metrics['after'] = {
            'ECE':after_temperature_ece.detach().cpu().numpy().item(), 
            'Brier':after_temperature_brier, 
            'NLL':after_temperature_nll
        }
    def save(self, save_name):
        torch.save(self.state_dict(), save_name)
        # save metrics
        m_fp = save_name.replace(".pt", "_metrics.json")
        print(self.temp_metrics)
        with open(m_fp, 'w') as fp:
            json.dump(self.temp_metrics, fp)
    
    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))




