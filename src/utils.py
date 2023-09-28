import torch
from torch import nn, optim
import os
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from collections import namedtuple
from scipy.stats import mannwhitneyu



class CalibratedModel():
    '''
    Temperature scaling for model calibration
    
    '''
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



# Copyright 2019 Brandon Walraven

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), 
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# inspired by paper "Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification"
# by Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, Lucy Vasserman.
# from
# https://github.com/bcwalraven/biasMetrics
class NuancedROC:
    """Method for calculating nuanced AUR ROC scores to assess model bias.
    Nuanced AUC ROC scores allow for a closer look into how a classification
    model performs across any specifed sub-population in the trainging set. 
    There are three different types of nuanced roc metrics included in this
    class.
    Subgroup (SG) ROC: 
    This calculates the AUC ROC score for only a specific subgroup of the 
    population. This value can be compared against the overall AUC ROC score
    for the entire population to see if the model underperforms or overperforms
    in classifying the subgroup in question.
    Background Positive Subgroup Negative (BPSN) ROC:
    This calculates the AUC ROC score for positive (relative to the target)
    members of the background (non-subgroup) population and negative members
    of the subgroup population. This value can be compared to see how the 
    model performs at differentiating between positive members on the background
    population and negative members of the subgroup population.  
    Background Negative Subgroup Positive (BNSP) ROC:
    This calculates the AUC ROC score for negative (relative to the target)
    members of the background (non-subgroup) population and positive members
    of the subgroup population. This value can be compared to see how the 
    model performs at differentiating between negative members on the background
    population and positive members of the subgroup population.  
    Read more about how to compare scores in "Nuanced Metrics for Measuring 
    Unintended Bias with Real Data for Text Classification" by Daniel Borkan, 
    Lucas Dixon, Jeffrey Sorensen, Nithum Thain, Lucy Vasserman.
    https://arxiv.org/abs/1903.04561
    Methods
    ----------
    score : Calculates nuanced roc scores for all given parameters and and returns 
            a heat map with the scores for each subpopulation.
    Attributes
    ----------
    
    mean_SG_roc : Returns the mean of the SG ROCs for all subgroups.
        
    mean_BPSN_roc : Returns the mean of the BPSN ROCs for all subgroups.
        
    mean_BNSP_roc : Returns the mean of the BNSP ROCs for all subgroups.
        
    mean_roc : Returns the weighted mean of the SG, BPSN, and BNSP scores
               for all specified subgroups. 
        
    summary : Prints out all the scores for each subgroup.
    """

    def __init__(self):
        self.output_df = pd.DataFrame()
        
        
    def score(self, y_true, y_probs, subgroup_df):
        """Parameters
        ----------
        y_true : pandas Series, pandas DataFrame
            The true values for all observations.
        y_pred : pandas Series, pandas DataFrame
            The model's predicted values for all observations.
        subgroup_df : pandas DataFrame
            Dataframe of all subgroups to be compared. Each column should be a
            specific subgroup with 1 to indicating the observation is a part of
            the subgroup and 0 indicating it is not. There should be no other values
            besides 1 or 0 in the dataframe."""

        def calc_SG_roc(parameter, df):
            SG = df.loc[df[parameter] == 1]
            SG_roc = roc_auc_score(y_true=SG.target, y_score=SG['probs'])
            return SG_roc

        # define functions to calculate specific ROC AUC for subpopulations within the data
        def calc_BPSN_roc(parameter, df):
            BPSN = df[((df.target == 1) & (df[parameter] == 0)) | ((df.target == 0) & (df[parameter] == 1))]
            BPSN_roc = roc_auc_score(y_true=BPSN.target, y_score=BPSN['probs'])
            return BPSN_roc

        def calc_BNSP_roc(parameter, df):
            BNSP = df[((df.target == 0) & (df[parameter] == 0)) | ((df.target == 1) & (df[parameter] == 1))]
            BNSP_roc = roc_auc_score(y_true=BNSP.target, y_score=BNSP['probs'])
            return BNSP_roc

        # ensure that the passed dataframe has an appropriate axis    
        subgroup_df.reset_index(drop=True, inplace=True)


        # ensure input true and prob values are formatted correctly
        if type(y_true) == pd.core.frame.DataFrame:
            y_true.columns = ['target']
            y_true.reset_index(drop=True, inplace=True)
        else:
            y_true = pd.DataFrame(y_true, columns=['target']).reset_index(drop=True)

        if type(y_probs) == pd.core.frame.DataFrame:
            y_probs.columns = ['probs']
            y_probs.reset_index(drop=True, inplace=True)
        else:
            y_probs = pd.DataFrame(y_probs, columns=['probs']).reset_index(drop=True)
            
        # combine all inputs into a DataFrame
        input_df = pd.concat([y_true, y_probs, subgroup_df], axis=1)

        # build dataframe and fill with ROC AUC metrics
        self.output_df = pd.DataFrame(index=subgroup_df.columns, columns=['SG-ROC', 'BPSN-ROC', 'BNSP-ROC'])
        for col in subgroup_df.columns:
            self.output_df.loc[col] = [calc_SG_roc(col, input_df), 
                                       calc_BPSN_roc(col, input_df), 
                                       calc_BNSP_roc(col, input_df)]

        self.model_roc = roc_auc_score(y_true=y_true, y_score=y_probs)

        self.mean_SG_roc = self.output_df['SG-ROC'].mean()

        self.mean_BPSN_roc = self.output_df['BPSN-ROC'].mean()

        self.mean_BNSP_roc = self.output_df['BNSP-ROC'].mean()

        self.mean_bias_roc = np.mean([self.output_df['SG-ROC'].mean(), 
                                      self.output_df['BPSN-ROC'].mean(), 
                                      self.output_df['BNSP-ROC'].mean()])
        return self.output_df





class AEG:
    """Method for calculating the Average Equality Gap (AEG) for true positive 
    rates (TPR) from a subpopulation and the background population to assess model 
    bias. AEG scores allow a closer look into how a binary classification model 
    performs across any specified subpopulation in the dataset. It compares how 
    the difference between TPR for a subpopulation the background population across 
    all probability thresholds. A perfectly balanced model will have a score of 0, 
    indicating there is no difference in the TPR between the two populations. A 
    total imbalance in the model will result in a score of 0.5 or -0.5, depending 
    on the direction of the skew. In this case all scores are interpreted relative 
    to the subpopulation. Positive scores indicate the model skews towards the 
    subpopulation and negative scores indicate the model skews away from the 
    subpopulation. 
    
    Conceptually this is difference between the curve of the rates (x(t)) and the 
    line y = x (y(t)) calculated as the integral (0, 1) of x(t) - y(t). This class 
    makes use of a simplified closed-form solution using the Mann Whitney U test. 
    There are two different AEG metrics included in this class.
    Positive AEG: 
    Calculates the average distance between the TPRs for all members of the 
    subpopulation and background population in the target class (1). Positive 
    scores indicate a rightward shift in the subpopulation and a tendency for the 
    model to produce false positives. Negative scores indicate a leftward shift in 
    the subpopulation and a tendency for the model to produce false negatives.
    Negative AEG:
    Calculates the average distance between the TPRs for all members of the 
    subpopulation and background population in the non-target class (0). Positive 
    scores indicate a rightward shift in the subpopulation and a tendency for the 
    model to produce false positives. Negative scores indicate a leftward shift in 
    the subpopulation and a tendency for the model to produce false negatives.
    Read more about how to compare scores in "Nuanced Metrics for Measuring 
    Unintended Bias with Real Data for Text Classification" by Daniel Borkan, 
    Lucas Dixon, Jeffrey Sorensen, Nithum Thain, Lucy Vasserman.
    https://arxiv.org/abs/1903.04561
    Methods
    ----------
    score : Calculates positive and negative AEG scores for all given parameters 
            and returns a heat map with the scores for each subpopulation.
    """

    def __init__(self):
        self.output_df = pd.DataFrame()
        
        
    def score(self, y_true, y_probs, subgroup_df):
        """Parameters
        ----------
        y_true : pandas Series, pandas DataFrame
            The true values for all observations.
        y_pred : pandas Series, pandas DataFrame
            The model's predicted values for all observations.
        subgroup_df : pandas DataFrame
            Dataframe of all subgroups to be compared. Each column should be a
            specific subgroup with 1 to indicating the observation is a part of
            the subgroup and 0 indicating it is not. There should be no other values
            besides 1 or 0 in the dataframe.
        output : boolean (default = True)
            If true returns a heatmap of the AEG scores.
        """

        def calc_pos_aeg(parameter, df): 
            sub_probs = df[((df.target == 1) & (df[parameter] == 1))]['probs']
            back_probs = df[((df.target == 1) & (df[parameter] == 0))]['probs']
            pos_aeg = (.5 - (mannwhitneyu(sub_probs, back_probs)[0] / (len(sub_probs)*len(back_probs))))
            return round(pos_aeg, 2) 
        
        def calc_neg_aeg(parameter, df): 
            sub_probs = df[((df.target == 0) & (df[parameter] == 1))]['probs']
            back_probs = df[((df.target == 0) & (df[parameter] == 0))]['probs']
            neg_aeg = (.5 - (mannwhitneyu(sub_probs, back_probs)[0] / (len(sub_probs)*len(back_probs))))
            return round(neg_aeg, 2) 

        # ensure that the passed dataframe has an appropriate axis    
        subgroup_df.reset_index(drop=True, inplace=True)


        # ensure input true and prob values are formatted correctly
        if type(y_true) == pd.core.frame.DataFrame:
            y_true.columns = ['target']
            y_true.reset_index(drop=True, inplace=True)
        else:
            y_true = pd.DataFrame(y_true, columns=['target']).reset_index(drop=True)

        if type(y_probs) == pd.core.frame.DataFrame:
            y_probs.columns = ['probs']
            y_probs.reset_index(drop=True, inplace=True)
        else:
            y_probs = pd.DataFrame(y_probs, columns=['probs']).reset_index(drop=True)
            
        # combine all inputs into a DataFrame
        input_df = pd.concat([y_true, y_probs, subgroup_df], axis=1)

        # build dataframe and fill with ROC AUC metrics
        self.output_df = pd.DataFrame(index=subgroup_df.columns, columns=['Positive AEG', 'Negative AEG'])
        for col in subgroup_df.columns:
            self.output_df.loc[col] = [calc_pos_aeg(col, input_df), 
                                       calc_neg_aeg(col, input_df)]
        return self.output_df
        
        
# MIT License

# Copyright (c) 2017 Geoff Pleiss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# modified from https://github.com/gpleiss/equalized_odds_and_calibration


class CalibEqOddsModel(namedtuple('CalibEqOddsModel', 'pred label')):
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label

    def calib_eq_odds(self, other, fp_rate, fn_rate, mix_rates=None):
        if mix_rates is None:
            if fn_rate == 0:
                self_cost = self.fp_cost()
                other_cost = other.fp_cost()
                print(self_cost, other_cost)
                self_trivial_cost = self.trivial().fp_cost()
                other_trivial_cost = other.trivial().fp_cost()
            elif fp_rate == 0:
                self_cost = self.fn_cost()
                other_cost = other.fn_cost()
                self_trivial_cost = self.trivial().fn_cost()
                other_trivial_cost = other.trivial().fn_cost()
            else:
                self_cost = self.weighted_cost(fp_rate, fn_rate)
                other_cost = other.weighted_cost(fp_rate, fn_rate)
                self_trivial_cost = self.trivial().weighted_cost(fp_rate, fn_rate)
                other_trivial_cost = other.trivial().weighted_cost(fp_rate, fn_rate)
    
            other_costs_more = other_cost > self_cost
            self_mix_rate = (other_cost - self_cost) / (self_trivial_cost - self_cost) if other_costs_more else 0
            other_mix_rate = 0 if other_costs_more else (self_cost - other_cost) / (other_trivial_cost - other_cost)
        else:
            self_mix_rate = mix_rates[0]
            other_mix_rate = mix_rates[1]

        # New classifiers
        self_indices = np.random.permutation(len(self.pred))[:int(self_mix_rate * len(self.pred))]
        self_new_pred = self.pred.copy()
        self_new_pred[self_indices] = self.base_rate()
        calib_eq_odds_self = CalibEqOddsModel(self_new_pred, self.label)

        other_indices = np.random.permutation(len(other.pred))[:int(other_mix_rate * len(other.pred))]
        other_new_pred = other.pred.copy()
        other_new_pred[other_indices] = other.base_rate()
        calib_eq_odds_other = CalibEqOddsModel(other_new_pred, other.label)

        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, [self_mix_rate, other_mix_rate]
        else:
            #return calib_eq_odds_self, calib_eq_odds_other
            return calib_eq_odds_self, calib_eq_odds_other

    def trivial(self):
        """
        Given a classifier, produces the trivial classifier
        (i.e. a model that just returns the base rate for every prediction)
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate
        return CalibEqOddsModel(pred, self.label)

    def weighted_cost(self, fp_rate, fn_rate):
        """
        Returns the weighted cost
        If fp_rate = 1 and fn_rate = 0, returns self.fp_cost
        If fp_rate = 0 and fn_rate = 1, returns self.fn_cost
        If fp_rate and fn_rate are nonzero, returns fp_rate * self.fp_cost * (1 - self.base_rate) +
            fn_rate * self.fn_cost * self.base_rate
        """
        norm_const = float(fp_rate + fn_rate) if (fp_rate != 0 and fn_rate != 0) else 1
        res = fp_rate / norm_const * self.fp_cost() * (1 - self.base_rate()) + \
            fn_rate / norm_const * self.fn_cost() * self.base_rate()
        return res