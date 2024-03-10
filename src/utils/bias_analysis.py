import argparse
import os
import pandas as pd
from scipy.special import logit, expit
import torch
import sklearn.metrics as sk_metrics
import numpy as np
import sys
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(os.path.join('..', 'src')))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.plot_formatting import *


def info_pred_mapping(info:pd.DataFrame, pred:pd.DataFrame)->pd.DataFrame:
    """ Map patient attributes information (e.g. sex, race) to prediction score and labels
    according to the patient id
    
    Arguments
    =========
    info
        Dataframe contains patient attributes info.
    pred
        Dataframe contains model prediction scores.

    Returns
    =======
    pandas.DataFrame
        Dataframe combines patient attributes and predictions.
    
    """
    # drop duplicate patient ids
    info.drop_duplicates(subset="patient_id", keep='first', inplace=True)
    # read prediction result file
    info_pred = pred.copy()
    # mapping patient labels to output score
    info_cols = [c for c in info.columns if c not in ['patient_id', 'Path']]
    for c in info_cols:
        info_pred[c] = info_pred['patient_id'].map(info.set_index("patient_id")[c])
    return info_pred
    

def metric_calculation(result_df, info_pred, test_list, positive_group, prev_diff, threshold=0.5):
    """ Calculate performance and bias measurements for subgroups. 
    
    Arguments
    =========
    info_pred
        Dataframe contains patient attributes and predictions.
    test_list
        List of subgroups for bias measurements calculation.
    output_file
        File path to store calculated performance and bias measurements.
    threshold
        Threshold

    Returns
    =======
    pandas.DataFrame
        Dataframe contains calculated performance and bias measurements.
    
    """
    for grp in test_list:
        dp = {}
        info_sub = info_pred.loc[info_pred[grp]==1]
        task_gt = info_sub['label']
        task_pred = info_sub['score']
        dp['subgroup'] = [grp]
        dp['experiment'] = [abs(prev_diff)]
        dp['positive associated'] = [positive_group]
        # Predicted prevalence
        dp['metric'] = ["Predicted Prevalence"]
        dp['value'] = len(info_sub[info_sub['score'] > 0.5]) / len(info_sub)
        result_df = pd.concat([result_df, pd.DataFrame(dp)], ignore_index=True)
        if prev_diff == 0:
            result_df = pd.concat([result_df, pd.DataFrame({**dp, 'positive associated': [test_list[1]]})], ignore_index=True)
        # AUROC
        dp['metric'] = ["AUROC"]
        dp['value'] = [sk_metrics.roc_auc_score(y_score=task_pred, y_true=task_gt)]
        result_df = pd.concat([result_df, pd.DataFrame(dp)], ignore_index=True)
        if prev_diff == 0:
            result_df = pd.concat([result_df, pd.DataFrame({**dp, 'positive associated': [test_list[1]]})], ignore_index=True)            
    return result_df
  
def analysis(args):
    """
    Main script to load test results, measure bias and do plotting.
    
    Arguments
    =========
    args : argparse.Namespace
        The input arguments to the python script.
    """
    # # get basic infomation for bias measurements
    main_dir = args.main_dir  
    test_info = pd.read_csv(args.testing_info_file)
    group_dict = {'sex':['F', 'M'], 'race':['Black', 'White']}    
    test_list = group_dict.get(args.test_subgroup)
    result_df = pd.DataFrame(columns=['subgroup', 'experiment', 'positive associated', 'metric', 'value'])    
    # # calculate bias measurements
    for exp in args.exp_list:        
        if args.amplification_type.lower() == "quantitative misrepresentation":            
            x_tick = 2 * int(''.join(filter(str.isdigit, exp))) - 100
            pos_as = test_list[0] if x_tick >= 0 else test_list[1]
            x_label = "Training Prevelance Difference(%)"
            y_lim = {'AUROC':[0.5, 0.75], 'Predicted Prevalence':[0,1]}
        elif args.amplification_type.lower() == "inductive transfer learning":
            x_tick = int(''.join(filter(str.isdigit, exp))) if "baseline" not in exp else 0
            pos_as = exp[0] if "baseline" not in exp else test_list[0]
            x_label = "Number of Frozen Layers"
            y_lim = {'AUROC':[0.5, 0.75], 'Predicted Prevalence':[0.25,0.75]}
        else:
            print('ERROR. UNKNOWN bias amplification type.')
            return
        test_score = pd.read_csv(os.path.join(main_dir, exp, args.testing_result_file), sep='\t')
        test_info_result = info_pred_mapping(test_info, test_score)        
        result_df = metric_calculation(result_df, test_info_result, test_list, pos_as, x_tick, args.threshold)
    # # plot the metric            
    results_plotting(data=result_df, x_col="experiment", x_label=x_label, hue_col="subgroup", style_col="positive associated",
                    s_col="metric", value_col="value", exp_type=args.amplification_type, y_lim=y_lim)
    
def results_plotting(data, x_col, x_label, hue_col, style_col, s_col, value_col, exp_type, color_dict=COLORS, style_dict=STYLES, y_lim=None):  
    """
    Generates subplots based on input plot sections and parameters.

    Arguments
    =========
    data
        dataframe that contains the data for plotting
    x_col
        name for column that contains x-axis ticks
    x_label
        set the x label name
    hue_col
        name for column that contains subgroups mapped with different colors during plotting
    style_col
        name for column that determine line styles by positive-associated subgroup
    s_col
        name for column that contains sub-sections in the figure    
    value_col
        name for column that contains metric value
    exp_type
        indicate which bias amplification approach
    style_dict
        dictionary that determines plotting style
    color_dict
        dictionary that determins plotting colors
    y_lim
        set range for y axis according to metric
    """
    # # create figure with sub-sections       
    fig = plt.figure(figsize = (10, 4))
    fig.suptitle(exp_type.title(), fontsize = 10, weight='bold')  
    gs = fig.add_gridspec(1, 2)
    data = data.sort_values(x_col)
    axes = []
    for c in range(2):
        axes.append(fig.add_subplot(gs[0,c]))
        axes[-1].set_xticks(data[x_col].unique().tolist())
        axes[-1].set_xlabel(x_label)
    # # generate plots in each sub-sections
    for i, m in enumerate(sorted(data[s_col].unique().tolist())):
      ax = axes[i]
      ax.set_title(f"Subgroup {m}")
      ax.set_ylabel(m)
      if y_lim is not None:
          ylim = y_lim.get(m)
          ax.set_ylim(ylim[0], ylim[1])
      temp_data = data[data[s_col] == m].copy()
      gb = [hue_col, style_col]
      for gp, df in temp_data.groupby(gb):
        hue = gp[0]
        style = gp[-1]
        ax.plot(df[x_col], df[value_col], c=color_dict[hue], ls=style_dict[style], linewidth=3)
        ax.set_xticks(data[x_col].unique().tolist())
        # set the "0" to "B" (for baseline)
        labels = ax.get_xticks().tolist()
        labels[0] = "B"
        ax.set_xticklabels(labels)
        name_map = SUBGROUP_NAME_MAPPING
        for h in data[hue_col].unique().tolist():
            if h not in name_map:
                name_map[h] = h
        hue_lines = [Patch(facecolor=color_dict[h], label=name_map[h]) for h in data[hue_col].unique().tolist()]
        hue_legend = ax.legend(handles=hue_lines,  title=hue_col, loc='lower left')
        if style_col:
          style_lines = [Line2D([0], [0], ls=style_dict[s], color='k', label=name_map[s]) for s in data[style_col].unique().tolist()]
          style_legend = ax.legend(handles=style_lines, title="Positive-Associated", loc='lower right')
          fig.add_artist(hue_legend)  
    # # show the figure
    plt.show()
    plt.close("all")               

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--main_dir',type=str, required=True)
    parser.add_argument('-a', '--amplification_type',type=str, required=True,
    help="Choose which bias amplification type (quantitative misrepresentation or inductive transfer learning.")
    parser.add_argument('-e', '--exp_list', nargs='+', default=[], required=True)
    parser.add_argument('-r', '--testing_result_file',type=str, required=True)
    parser.add_argument('-i', '--testing_info_file',type=str, required=True)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-s', '--test_subgroup', type=str, required=True)
    args = parser.parse_args()
    print("\nStart subgroup bias measurements")
    analysis(args)    
    print("\nDone\n")

