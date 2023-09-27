import torch
from torch import nn, optim
from torch.nn import functional as F
import os
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss
import json

class CalibratedModel():
    def __init__(self, calibration_mode='temperature'):
        super(CalibratedModel, self).__init__()
        self.calibration_mode = calibration_mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.calibration_mode == 'temperature':
            self.temperature = nn.Parameter(torch.ones(1)*1.5)

    def temperature_scale(self, logits):
        temperature = self.temperature.expand(logits.size(0)).to(self.device)
        return logits / temperature

    def get_metrics(self, logits, labels):
        '''Returns AUROC, Brier, ECE, and NLL scores for binary classification tasks'''
        brier = brier_score_loss(labels, expit(logits), pos_label=1)
        # convert to tensors
        p = torch.tensor(logits)
        l = torch.tensor(labels)
        ece_criterion = _ECELoss().to(self.device)
        ece = ece_criterion(p, l).item()
        nll_criterion = nn.BCEWithLogitsLoss().to(self.device)
        nll = nll_criterion(p, l.double()).item()
        out = pd.DataFrame({'Brier':[brier], 'NLL':[nll], 'ECE':[ece]})
        return out

    def set_temperature(self, valid_results, lr=0.01, max_iter=50):
        self.temperature.to(self.device)
        self.temp_metrics = {}
        # NOTE: Brier score only works for binary classificaiton, assumes first column is positive percentage
        self.temp_lr = lr
        self.temp_max_iter = max_iter
        #nll_criterion = nn.CrossEntropyLoss().cuda()
        nll_criterion = nn.BCEWithLogitsLoss().to(self.device)
        # expects valid_predictions = [preds, labels]
        if len([valid_results[1].shape]) != 1:
            labels = torch.argmax(valid_results[1], dim=1).to(self.device)
        else:
            labels = torch.from_numpy(valid_results[1]).to(self.device)
        logits = torch.from_numpy(valid_results[0]).to(self.device)
        if len([logits.shape]) == 1:
            # binary classification
            nll_criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.LBFGS([self.temperature], lr=self.temp_lr, max_iter=self.temp_max_iter)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels.to(torch.float))
            loss.backward()
            return loss
        optimizer.step(eval)
        return self.temperature.detach().cpu().numpy()


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        scores = torch.sigmoid(logits)
        confidences, predictions = torch.max(scores, 0)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece



