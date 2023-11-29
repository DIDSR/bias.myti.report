from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager

import cv2

plt.rcParams["figure.figsize"] = [8, 6]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'monospace'
plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.titley'] = 1.0 
plt.rcParams['axes.titlepad'] = -18
plt.rcParams["legend.loc"] = 'lower right'
plt.rcParams["legend.handlelength"] = 4.5
plt.rcParams["legend.borderaxespad"] = 0.2
plt.rcParams["legend.fontsize"] = 14
font = font_manager.FontProperties(weight='bold')

def result_plotting(sub, metric, exp_1, exp_2, csv_path):
    if not os.path.exists('../example/tmp/'):
        os.makedirs('../example/tmp/')
    data = pd.read_csv(csv_path)
    sub_list = list(data[sub].unique())
    sub_1 = sub_list[0]
    sub_2 = sub_list[1]
    data_1 = data.loc[data[sub] == sub_1]
    data_2 = data.loc[data[sub] == sub_2]
    x = np.arange(0, data_1.shape[0], 1, dtype=int)
    xticks = data_1[exp_1].tolist()
    
    
    scores_1 = data_1[metric].to_numpy()
    scores_2 = data_2[metric].to_numpy()
        
    fig, ax = plt.subplots()
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_xlabel('Disease Prevalence in Training Set', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.plot(x, scores_1, color='orange', linewidth=3)
    ax.plot(x, scores_2[::-1], color='#1f77b4', linewidth=3)
    ax.grid(color = "lightgray")
    plt.axvline(3, ymax = 0.92, color='navy',label = "Baseline", linestyle = '--', linewidth=3)
    plt.text(2.72, 0.02, "Baseline", rotation=90, weight="bold", color='navy')
    ax.legend([sub_1, sub_2], prop=font)
    ax.set_title(metric, fontweight='bold', bbox=dict(facecolor='khaki', alpha=0.2, edgecolor=(0,0,0,1)))
    fig.tight_layout()
    plt.savefig('../example/example_1.png')
    text = f'The plot shows the effect of subgroup disease prevalence\n in the training set on the {metric} in the test set.\n'+ \
    f'The baseline at 50% show that the {metric} difference\n is at {abs(scores_1[4]-scores_2[4])} when data is well balanced.'
    with open("../example/tmp/description_1.txt", "w") as f:
        f.write(text)
    
    fig, ax = plt.subplots()
    ax.set_xticks(x)
    ax.set_xticklabels(xticks)
    ax.set_xlabel(f'Disease Prevalence {sub_1}', fontweight='bold')
    ax.set_ylabel('Predicted Prevalence', fontweight='bold')
    ax.set_ylim(0, 1)
    ax.plot(x, scores_1, color='orange', linewidth=3)
    ax.plot(x, scores_2, color='#1f77b4', linewidth=3)
    ax.grid(color = "lightgray")
    plt.axvline(3, ymax = 0.92, color='navy',label = "Baseline", linestyle = '--', linewidth=3)
    plt.text(2.72, 0.02, "Baseline", rotation=90, weight="bold", color='navy')
    ax.legend([sub_1, sub_2], prop=font)
    ax.set_title(metric, fontweight='bold', bbox=dict(facecolor='khaki', alpha=0.2, edgecolor=(0,0,0,1)))
    plt.savefig('../example/example_2.png')
    text = f'The plot emphasizes the effect of subgroup disease prevalence difference\n in the training set on the subgroup {metric} difference in the test set.\n' + \
    f'The baseline at 50% show that the {metric} difference\n is at {abs(scores_1[4]-scores_2[4])} when data is well balanced.\n' + \
    f'The extreme case at 0% and 100% has\n the {metric} difference of {abs(scores_1[0]-scores_2[0])}  and {abs(scores_1[-1]-scores_2[-1])}.'
    with open("../example/tmp/description_2.txt", "w") as f:
        f.write(text)
    
    x_2 = abs(data_1[exp_1].to_numpy() - data_1[exp_2].to_numpy())[-4:]
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_2)
    #ax.set_xticklabels(xticks)
    ax.set_xlabel('Disease Prevalence Difference in Training Set', fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.plot(x_2, scores_1[3::-1], color='orange', linewidth=3)
    ax.plot(x_2, scores_1[-4:], color='orange', linewidth=3, ls='--')
    ax.plot(x_2, scores_2[3::-1], color='#1f77b4', linewidth=3)
    ax.plot(x_2, scores_2[-4:], color='#1f77b4', linewidth=3, ls='--')
    ax.grid(color = "lightgray")
    plt.axvline(0, ymax = 1, color='navy',label = "Baseline", linestyle = '--', linewidth=3)
    plt.text(2.72, 0.02, "Baseline", rotation=90, weight="bold", color='navy')
    ax.legend([f'{sub_1} (positive associate {sub_2})', f'{sub_1} (positive associate {sub_1})', f'{sub_2} (positive associate {sub_2})', f'{sub_2} (positive associate {sub_1})'], prop=font)
    ax.set_title(metric, fontweight='bold', bbox=dict(facecolor='khaki', alpha=0.2, edgecolor=(0,0,0,1)))
    fig.tight_layout()
    plt.savefig('../example/example_3.png')
    text = f'The plot emphasizes the difference in subgroup predicted prevalence.'
    with open("../example/tmp/description_3.txt", "w") as f:
        f.write(text)
    

def save_fig_text(index, fname):
    plot = cv2.imread(f'../example/example_{index}.png')
    text_img = cv2.imread(f'../example/tmp/text.png')

    
    comb_img = cv2.vconcat([plot, text_img])
    cv2.imwrite(fname, comb_img)  
    
       
    
    