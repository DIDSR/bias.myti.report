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
import numpy as np
import pandas as pd
import argparse
import os
from collections import namedtuple


class Model(namedtuple('Model', 'pred label')):
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

        # New classifiers
        self_indices = np.random.permutation(len(self.pred))[:int(self_mix_rate * len(self.pred))]
        self_new_pred = self.pred.copy()
        self_new_pred[self_indices] = self.base_rate()
        calib_eq_odds_self = Model(self_new_pred, self.label)

        other_indices = np.random.permutation(len(other.pred))[:int(other_mix_rate * len(other.pred))]
        other_new_pred = other.pred.copy()
        other_new_pred[other_indices] = other.base_rate()
        calib_eq_odds_other = Model(other_new_pred, other.label)

        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, (self_mix_rate, other_mix_rate)
        else:
            #return calib_eq_odds_self, calib_eq_odds_other
            return self_new_pred, self.label, other_new_pred, other.label

    def trivial(self):
        """
        Given a classifier, produces the trivial classifier
        (i.e. a model that just returns the base rate for every prediction)
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate
        return Model(pred, self.label)

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

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])

if __name__ == '__main__':

    import pandas as pd
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--constraints',type=str)
    parser.add_argument('--constraint_weight',type=float,default=1)
    parser.add_argument('--test_dir',type=str)
    parser.add_argument('--train_file',type=str)
    parser.add_argument('--deploy_file',type=str)
    parser.add_argument('--train_info',type=str)
    parser.add_argument('--deploy_info',type=str)
    parser.add_argument('--output_file',type=str)
    parser.add_argument('--test_subgroup',type=str)
    args = parser.parse_args()
    test_dir = args.test_dir
    weight = args.constraint_weight
    
    custom_subgroups ={ 
    'sex':['M','F'],
    'race':['Black', 'White'],
    'COVID_positive':['Yes', 'No'],
    'modality':['CR', 'DX']
    }
    test_list = custom_subgroups.get(args.test_subgroup)
    print(test_list)
    #load prediction files and their attribute information
    validation_list = pd.read_csv(args.train_info)
    validation_list.drop_duplicates(subset="patient_id", keep='first', inplace=True)
    predictions_val = pd.read_csv(os.path.join(test_dir, args.train_file))
    info_pred_val = predictions_val.copy()
    info_cols = ['F','M','Black', "White", "Yes", 'No']
    cols = [c for c in info_pred_val.columns if c not in ['patient_id']]
    info_pred_val = info_pred_val.rename(columns={c:f"{c} score" for c in cols})
    for c in info_cols:
        info_pred_val[c] = info_pred_val['patient_id'].map(validation_list.set_index("patient_id")[c])
    
    validation_2_list = pd.read_csv(args.deploy_info)
    validation_2_list.drop_duplicates(subset="patient_id", keep='first', inplace=True)
    predictions = pd.read_csv(os.path.join(test_dir,args.deploy_file))
    info_pred = predictions.copy()
    info_pred = info_pred.rename(columns={c:f"{c} score" for c in cols})
    for c in info_cols:
        info_pred[c] = info_pred['patient_id'].map(validation_2_list.set_index("patient_id")[c])
    
    # Cost constraint
    cost_constraint = args.constraints
    if cost_constraint not in ['fnr', 'fpr', 'weighted']:
        raise RuntimeError('cost_constraint (arg #2) should be one of fnr, fpr, weighted')

    if cost_constraint == 'fnr':
        fn_rate = 1
        fp_rate = 0
    elif cost_constraint == 'fpr':
        fn_rate = 0
        fp_rate = 1
    elif cost_constraint == 'weighted':
        fn_rate = 1
        fp_rate = weight

    # Load the validation set scores from csvs
    val_data = info_pred_val
    test_data = info_pred

    # Create model objects - one for each group, validation and test 
    group_0_val_data = val_data[val_data[test_list[0]] == 1]
    group_1_val_data = val_data[val_data[test_list[0]] == 0]   
    group_0_test_data = test_data[test_data[test_list[0]] == 1]
    group_1_test_data = test_data[test_data[test_list[0]] == 0]
    
    group_0_val_model = Model(group_0_val_data['Yes score'].as_matrix(), group_0_val_data['Yes'].as_matrix())
    group_1_val_model = Model(group_1_val_data['Yes score'].as_matrix(), group_1_val_data['Yes'].as_matrix())
    group_0_test_model = Model(group_0_test_data['Yes score'].as_matrix(), group_0_test_data['Yes'].as_matrix())
    group_1_test_model = Model(group_1_test_data['Yes score'].as_matrix(), group_1_test_data['Yes'].as_matrix())
    

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.calib_eq_odds(group_0_val_model, group_1_val_model, fp_rate, fn_rate)

    # Apply the mixing rates to the test models
    calib_eq_odds_group_0_test_model, label_0, calib_eq_odds_group_1_test_model, label_1 = Model.calib_eq_odds(group_0_test_model,
                                                                                             group_1_test_model,
                                                                                             fp_rate, fn_rate,
                                                                                             mix_rates)
    
    # Reorganize the prediction scores and output as a new file
    pred = pd.DataFrame(calib_eq_odds_group_0_test_model, columns=['Yes score'])
    pred['Yes'] = label_0
    pred[test_list[0]] = '1'
    pred[test_list[1]] = '0'    
    pred_new = pd.DataFrame(calib_eq_odds_group_1_test_model, columns=['Yes score'])
    pred_new['Yes'] = label_1
    pred_new[test_list[0]] = '0'
    pred_new[test_list[1]] = '1'    
    pred_comb = pd.concat([pred, pred_new], axis=0)
    pred_comb.to_csv(os.path.join(test_dir, args.output_file), index=False)
