import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
from torchmetrics.functional import calibration_error
from sklearn.metrics import brier_score_loss
import json

val_prediction_file = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/1_step_all_CR_stratified_ind_test_BETSY/RAND_10/VALIDATION_SIZE_0__step_0/results/100_equal__validation__predictions_and_groundtruth.csv"
subgroups = ['FW', 'FB', 'MW', 'MB']


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
        return calibration_error(target=y_true, preds=y_pred, norm='l1')
    
    def brier(self, y_true, y_prob, pos_label=1):
        return brier_score_loss(y_true.cpu().detach().numpy(), y_prob.cpu().detach().numpy(), pos_label=pos_label)

    def set_temperature(self, valid_loader=None, valid_results=None, lr=0.01, max_iter=50):
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
            labels = torch.argmax(valid_results[1], dim=1).cuda()
            logits = valid_results[0].cuda()
        # calculate before-scaling metrics
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = self.ece(labels, logits)
        before_temperature_brier = self.brier(labels, logits[:,0], pos_label=0)# needs to be changed depending on exact model task set up

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
        after_temperature_brier = self.brier(labels, self.temperature_scale(logits)[:,0], pos_label=0)
        ch_brier = after_temperature_brier - before_temperature_brier
        print(f"\nAfter temperature scaling:\nNLL: {after_temperature_nll:.4f} ({ch_nll})\nECE: {after_temperature_ece:.4f} ({ch_ece})\nBrier: {after_temperature_brier:.4f} ({ch_brier})")
        print(f"Temperature: {self.temperature.item():.4f}")
        print(f"(max_iter={self.temp_max_iter}, lr={self.temp_lr})\n")

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)
        # atts = vars(self)
        # with open(save_name, 'w') as fp:
        #     json.dump(atts, fp, indent=2)
    
    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))
        # with open(file_name, 'w') as fp:
        #     atts = json.load(fp)
        # for key in atts:
        #     setattr(self, key, atts[key])




