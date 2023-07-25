from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import argparse

plt.rcParams["figure.figsize"] = [9.6, 7.2]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'monospace'
plt.rcParams.update({'font.size': 16})

def plot_indirect_sex(args):
    # #approach indirect sex
    x_ind = np.array([0,1,2,3,4])
    xticks_ind = ['Baseline',	'1', '12',	'14',	'15'] 
    #subgroup auroc
    auroc_1_mean_n = np.array([0.784157948,	0.773529846,	0.760813257,	0.74550136,	0.703751463])
    auroc_1_err_n = np.array([0.004004933,	0.00538384,	0.009680493,	0.013998313,	0.02050655])
    auroc_1_mean_i = np.array([0.784157948,	0.765534297,	0.764011917,	0.739362395,	0.685738284])
    auroc_1_err_i = np.array([0.004004933,	0.011580494,	0.017152214,	0.013268327,	0.011841107])
    auroc_2_mean_n = np.array([0.754730457,	0.747407326,	0.732896714,	0.702224723,	0.659590969])
    auroc_2_err_n = np.array([0.010221769,	0.002297059,	0.007571902,	0.011271781,	0.004597261])
    auroc_2_mean_i = np.array([0.754730457,	0.741604781,	0.728706034,	0.700802492,	0.653543851])
    auroc_2_err_i = np.array([0.010221769,	0.008288886,	0.011838035,	0.012822463,	0.010740053])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_ind)
    ax.set_xticklabels(xticks_ind)
    ax.set_xlabel('Number of frozen layers')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    ax.plot(x_ind, auroc_1_mean_n, color='orange', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, auroc_1_mean_i, color='orange', linestyle = 'dashed', linewidth=3)
    ax.plot(x_ind, auroc_2_mean_n, color='#1f77b4', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, auroc_2_mean_i, color='#1f77b4', linestyle = 'dashed', linewidth=3)
    ax.fill_between(x_ind, auroc_1_mean_n - auroc_1_err_n, auroc_1_mean_n + auroc_1_err_n, facecolor ='orange', alpha=0.2)
    ax.fill_between(x_ind, auroc_1_mean_i - auroc_1_err_i, auroc_1_mean_i + auroc_1_err_i, facecolor ='orange', alpha=0.2)
    ax.fill_between(x_ind, auroc_2_mean_n - auroc_2_err_n, auroc_2_mean_n + auroc_2_err_n, facecolor ='#1f77b4', alpha=0.2)
    ax.fill_between(x_ind, auroc_2_mean_i - auroc_2_err_i, auroc_2_mean_i + auroc_2_err_i, facecolor ='#1f77b4', alpha=0.2)
    plt.legend(['Female (Class "1": Female)', 'Female (Class "1": Male)', 'Male (Class "1": Female)', 'Male (Class "1": Male)'], loc = 'lower right')
    plt.title("Indirect Approach-Subgroup AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_indirect_sex_auroc_subgroup.png"))
        
    #overall auroc
    auroc_all_mean_n = np.array([0.769152854,	0.75880341,	0.745605015,	0.723371897,	0.681888117])
    auroc_all_err_n = np.array([0.010700492,	0.009237161,	0.006161202,	0.006577065,	0.014675036])
    auroc_all_mean_i = np.array([0.769152854,	0.752178927,	0.745671322,	0.720152834,	0.668801785])
    auroc_all_err_i = np.array([0.010700492,	0.011067353,	0.01279798,	0.01109494,	0.010416997])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_ind)
    ax.set_xticklabels(xticks_ind)
    ax.set_xlabel('Number of frozen layers')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    ax.plot(x_ind, auroc_all_mean_n, color='orangered', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, auroc_all_mean_i, color='orangered', linestyle = 'dashed', linewidth=3)
    ax.fill_between(x_ind, auroc_all_mean_n - auroc_all_err_n, auroc_all_mean_n + auroc_all_err_n, facecolor ='orangered', alpha=0.2)
    ax.fill_between(x_ind, auroc_all_mean_i - auroc_all_err_i, auroc_all_mean_i + auroc_all_err_i, facecolor ='orangered', alpha=0.2)
    plt.legend(['Class "1": Female', 'Class "1": Male'], loc = 'lower right')
    plt.title("Indirect Approach-Overall AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_indirect_sex_auroc_overall.png"))
        
    #sensitivity
    tpr_1_mean_n = np.array([0.705851419,	0.707681485,	0.720064477,	0.737676206,	0.7421772])
    tpr_1_err_n = np.array([0.029295834,	0.024525018,	0.025974967,	0.028209761,	0.048661103])
    tpr_2_mean_n = np.array([0.667752732,	0.620959699,	0.614494536,	0.614560792,	0.566615437])
    tpr_2_err_n = np.array([0.042563048,	0.027886956,	0.02874044,	0.028853842,	0.033355279])
    tpr_1_mean_i = np.array([0.705851419, 0.685320052,	0.673897374,	0.644651413,	0.641545144])
    tpr_1_err_i = np.array([0.029295834,	0.033923969,	0.03260751,	0.030833206,	0.036004581])
    tpr_2_mean_i = np.array([0.667752732, 0.672086749,	0.682424863,	0.687281421,	0.698213798])
    tpr_2_err_i = np.array([0.042563048,	0.030857412,	0.027309003,	0.034073445,	0.046896402])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_ind)
    ax.set_xticklabels(xticks_ind)
    ax.set_xlabel('Number of frozen layers')
    ax.set_ylabel('Sensitivity', fontweight='bold')
    plt.ylim(0, 1)
    ax.plot(x_ind, tpr_1_mean_n, color='orange', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, tpr_1_mean_i, color='orange', linestyle = 'dashed', linewidth=3)
    ax.plot(x_ind, tpr_2_mean_n, color='#1f77b4', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, tpr_2_mean_i, color='#1f77b4', linestyle = 'dashed', linewidth=3)
    ax.fill_between(x_ind, tpr_1_mean_n - tpr_1_err_n, tpr_1_mean_n + tpr_1_err_n, facecolor ='orange', alpha=0.2)
    ax.fill_between(x_ind, tpr_1_mean_i - tpr_1_err_i, tpr_1_mean_i + tpr_1_err_i, facecolor ='orange', alpha=0.2)
    ax.fill_between(x_ind, tpr_2_mean_n - tpr_2_err_n, tpr_2_mean_n + tpr_2_err_n, facecolor ='#1f77b4', alpha=0.2)
    ax.fill_between(x_ind, tpr_2_mean_i - tpr_2_err_i, tpr_2_mean_i + tpr_2_err_i, facecolor ='#1f77b4', alpha=0.2)
    plt.legend(['Female (Class "1": Female)', 'Female (Class "1": Male)', 'Male (Class "1": Female)', 'Male (Class "1": Male)'], loc = 'lower right')
    plt.title("Indirect Approach-Subgroup Sensitivity", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_indirect_sex_sensitivity_subgroup.png"))
    
    
def plot_indirect_race(args):   
    # #approach indirect race
    x_ind = np.array([0,1,2,3,4])
    xticks_ind = ['Baseline',	'1', '12',	'14',	'15'] 
    #subgroup auroc
    auroc_1_mean_n = np.array([0.727094841,	0.70888685,	0.685511443,	0.672880126,	0.657301106])
    auroc_1_err_n = np.array([0.023167162,	0.029170952,	0.019630987,	0.013413202,	0.014777122])
    auroc_1_mean_i = np.array([0.727094841,	0.707849029,	0.70111443,	0.686138404,	0.65969279])
    auroc_1_err_i = np.array([0.023167162,	0.014132357,	0.020661258,	0.02435037,	0.0254753])
    auroc_2_mean_n = np.array([0.827282764,	0.821811736,	0.812910794,	0.802303529,	0.781799901])
    auroc_2_err_n = np.array([0.018595831,	0.019400445,	0.020103839,	0.021438251,	0.027283684])
    auroc_2_mean_i = np.array([0.827282764,	0.815895002,	0.813446901,	0.808791274,	0.781940529])
    auroc_2_err_i = np.array([0.018595831,	0.01560508,	0.011567643,	0.010968749,	0.019102506])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_ind)
    ax.set_xticklabels(xticks_ind)
    ax.set_xlabel('Number of frozen layers')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    ax.plot(x_ind, auroc_1_mean_n, color='slategray', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, auroc_1_mean_i, color='slategray', linestyle = 'dashed', linewidth=3)
    ax.plot(x_ind, auroc_2_mean_n, color='green', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, auroc_2_mean_i, color='green', linestyle = 'dashed', linewidth=3)
    ax.fill_between(x_ind, auroc_1_mean_n - auroc_1_err_n, auroc_1_mean_n + auroc_1_err_n, facecolor ='slategray', alpha=0.2)
    ax.fill_between(x_ind, auroc_1_mean_i - auroc_1_err_i, auroc_1_mean_i + auroc_1_err_i, facecolor ='slategray', alpha=0.2)
    ax.fill_between(x_ind, auroc_2_mean_n - auroc_2_err_n, auroc_2_mean_n + auroc_2_err_n, facecolor ='green', alpha=0.2)
    ax.fill_between(x_ind, auroc_2_mean_i - auroc_2_err_i, auroc_2_mean_i + auroc_2_err_i, facecolor ='green', alpha=0.2)
    plt.legend(['Black (Class "1": Black)', 'Black (Class "1": White)', 'White (Class "1": Black)', 'White (Class "1": White)'], loc = 'lower right')
    plt.title("Indirect Approach-Subgroup AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_indirect_race_auroc_subgroup.png"))
    
    #overall auroc
    auroc_all_mean_n = np.array([0.776030739,	0.768441343,	0.763868026,	0.752561635,	0.720947472])
    auroc_all_err_n = np.array([0.010049666,	0.008402921,	0.009731451,	0.010201902,	0.010630521])
    auroc_all_mean_i = np.array([0.776030739,	0.759518148,	0.749083443,	0.741481297,	0.714566452])
    auroc_all_err_i = np.array([0.010049666,	0.012281408,	0.01200631,	0.012467079,	0.01608643])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_ind)
    ax.set_xticklabels(xticks_ind)
    ax.set_xlabel('Number of frozen layers')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    ax.plot(x_ind, auroc_all_mean_n, color='orangered', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, auroc_all_mean_i, color='orangered', linestyle = 'dashed', linewidth=3)
    ax.fill_between(x_ind, auroc_all_mean_n - auroc_all_err_n, auroc_all_mean_n + auroc_all_err_n, facecolor ='orangered', alpha=0.2)
    ax.fill_between(x_ind, auroc_all_mean_i - auroc_all_err_i, auroc_all_mean_i + auroc_all_err_i, facecolor ='orangered', alpha=0.2)
    plt.legend(['Class "1": Black', 'Class "1": White'], loc = 'lower right')
    plt.title("Indirect Approach-Overall AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_indirect_race_auroc_overall.png"))
    
    #sensitivity
    tpr_1_mean_n = np.array([0.839335964,	0.858960005,	0.869791811,	0.880447131,	0.90184426])
    tpr_1_err_n = np.array([0.033533148,	0.031921101,	0.03226068,	0.022391301,	0.029339859])
    tpr_2_mean_n = np.array([0.670579743,	0.675576192,	0.661393723,	0.65429006,	0.643765078])
    tpr_2_err_n = np.array([0.021071551,	0.027432102,	0.019608415,	0.019343356,	0.025011912])
    tpr_1_mean_i = np.array([0.839335964, 0.835871601,	0.811035891,	0.786891828,	0.769098725])
    tpr_1_err_i = np.array([0.033533148,	0.024869457,	0.019758671,	0.022649245,	0.025337798])
    tpr_2_mean_i = np.array([0.670579743, 0.673349972,	0.680629417,	0.685161111,	0.693455152])
    tpr_2_err_i = np.array([0.021071551,	0.023902558,	0.024330853,	0.028425781,	0.035633823])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_ind)
    ax.set_xticklabels(xticks_ind)
    ax.set_xlabel('Number of frozen layers')
    ax.set_ylabel('Sensitivity', fontweight='bold')
    plt.ylim(0, 1)
    ax.plot(x_ind, tpr_1_mean_n, color='slategray', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, tpr_1_mean_i, color='slategray', linestyle = 'dashed', linewidth=3)
    ax.plot(x_ind, tpr_2_mean_n, color='green', linestyle = 'solid', linewidth=3)
    ax.plot(x_ind, tpr_2_mean_i, color='green', linestyle = 'dashed', linewidth=3)
    ax.fill_between(x_ind, tpr_1_mean_n - tpr_1_err_n, tpr_1_mean_n + tpr_1_err_n, facecolor ='slategray', alpha=0.2)
    ax.fill_between(x_ind, tpr_1_mean_i - tpr_1_err_i, tpr_1_mean_i + tpr_1_err_i, facecolor ='slategray', alpha=0.2)
    ax.fill_between(x_ind, tpr_2_mean_n - tpr_2_err_n, tpr_2_mean_n + tpr_2_err_n, facecolor ='green', alpha=0.2)
    ax.fill_between(x_ind, tpr_2_mean_i - tpr_2_err_i, tpr_2_mean_i + tpr_2_err_i, facecolor ='green', alpha=0.2)
    plt.legend(['Black (Class "1": Black)', 'Black (Class "1": White)', 'White (Class "1": Black)', 'White (Class "1": White)'], loc = 'lower right')
    plt.title("Indirect Approach-Subgroup Sensitivity", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_indirect_race_sensitivity_subgroup.png"))
    
def plot_direct_sex(args):    
    # # approach direct sex
    x_d = np.array([0,1,2,3,4,5,6])
    xticks_d = ['0%','10%', '25%','50%','75%','90%','100%']    
    #subgroup auroc
    auroc_mean_1 = np.array([0.584054178, 0.630701905, 0.670369605, 0.693609963, 0.666540404, 0.624360652, 0.578440657])
    auroc_err_1 = np.array([0.022763642, 0.021952412, 0.021404695, 0.014887414, 0.019035371, 0.017131701, 0.018844106])
    auroc_mean_2 = np.array([0.612333563, 0.6597073, 0.690494146, 0.707505739, 0.68190714, 0.646712006, 0.601537534])
    auroc_err_2 = np.array([0.027135631, 0.024042334, 0.025918695, 0.0202698, 0.020218353, 0.022451871, 0.021724965])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_d)
    ax.set_xticklabels(xticks_d)
    ax.set_xlabel('Disease Prevalence (Female)', fontweight='bold')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    secx = ax.secondary_xaxis('top', functions=(lambda x: 6-x, lambda x: 6-x))
    secx.set_xticklabels(['0%', '0%','10%', '25%','50%','75%','90%','100%'])
    secx.set_xlabel('Disease Prevalence (Male)', fontweight='bold')
    ax.plot(x_d, auroc_mean_1, color='orange', linewidth=3)
    ax.plot(x_d, auroc_mean_2, color='#1f77b4', linewidth=3)
    ax.fill_between(x_d, auroc_mean_1 - auroc_err_1, auroc_mean_1 + auroc_err_1, facecolor ='orange', alpha=0.2)
    ax.fill_between(x_d, auroc_mean_2 - auroc_err_2, auroc_mean_2 + auroc_err_2, facecolor ='#1f77b4', alpha=0.2)
    plt.legend(['Female', 'Male'], loc = 'lower right')
    plt.title("Direct Approach-Subgroup AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_direct_sex_auroc_subgroup.png"))
    
    
    #auroc for sex classify
    auroc_mean = np.array([0.934883351, 0.880711375, 0.768007346, 0.511377554, 0.76465493, 0.87834639, 0.935146063])
    auroc_err = np.array([0.007041461, 0.014348164, 0.021854283, 0.024350596, 0.017189179, 0.014568729, 0.007291454])
    #auroc for covid classify
    auroc_mean_all = np.array([0.56415978, 0.606642705, 0.657309458, 0.700609074, 0.653527175, 0.601170655, 0.558671947])
    auroc_err_all = np.array([0.011993987, 0.013789042, 0.017646302, 0.013921423, 0.011104586, 0.009276943, 0.009162986])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_d)
    ax.set_xticklabels(xticks_d)
    ax.set_xlabel('Disease Prevalence (Female)', fontweight='bold')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    secx = ax.secondary_xaxis('top', functions=(lambda x: 6-x, lambda x: 6-x))
    secx.set_xticklabels(['0%', '0%','10%', '25%','50%','75%','90%','100%'])
    secx.set_xlabel('Disease Prevalence (Male)', fontweight='bold')
    ax.plot(x_d, auroc_mean, color='purple', linewidth=3)
    ax.plot(x_d, auroc_mean_all, color='orangered', linewidth=3)
    ax.fill_between(x_d, auroc_mean - auroc_err, auroc_mean + auroc_err, facecolor ='purple', alpha=0.2)
    ax.fill_between(x_d, auroc_mean_all - auroc_err_all, auroc_mean_all + auroc_err_all, facecolor ='orangered', alpha=0.2)
    plt.legend(['Patient Sex', 'COVID Status'], title="Measurement of separation", loc = 'lower right')
    plt.title("Direct Approach-Overall AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_direct_sex_auroc_overall.png"))
    
    #sensitivity
    tpr_mean_1 = np.array([0.188939394, 0.267575758, 0.393030303, 0.646969697, 0.814848485, 0.863484848, 0.892878788])
    tpr_err_1 = np.array([0.024212813, 0.03261904, 0.046369681, 0.036150602, 0.02797052, 0.026140023, 0.018637749])
    tpr_mean_2 = np.array([0.906515152, 0.867121212, 0.798939394, 0.659848485, 0.432575758, 0.272878788, 0.175606061])
    tpr_err_2 = np.array([0.020183862, 0.021976227, 0.027596112, 0.031082156, 0.040102052, 0.032830954, 0.023141222])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_d)
    ax.set_xticklabels(xticks_d)
    ax.set_xlabel('Disease Prevalence (Female)', fontweight='bold')
    ax.set_ylabel('Sensitivity', fontweight='bold')
    plt.ylim(0, 1)
    secx = ax.secondary_xaxis('top', functions=(lambda x: 6-x, lambda x: 6-x))
    secx.set_xticklabels(['0%', '0%','10%', '25%','50%','75%','90%','100%'])
    secx.set_xlabel('Disease Prevalence (Male)', fontweight='bold')
    ax.plot(x_d, tpr_mean_1, color='orange', linewidth=3)
    ax.plot(x_d, tpr_mean_2, color='#1f77b4', linewidth=3)
    ax.fill_between(x_d, tpr_mean_1 - tpr_err_1, tpr_mean_1 + tpr_err_1, facecolor ='orange', alpha=0.2)
    ax.fill_between(x_d, tpr_mean_2 - tpr_err_2, tpr_mean_2 + tpr_err_2, facecolor ='#1f77b4', alpha=0.2)
    plt.legend(['Female', 'Male'], loc = 'lower right')
    plt.title("Direct Approach-Subgroup Sensitivity", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_direct_sex_sensitivity_subgroup.png"))
    
def plot_direct_race(args):    
    # # approach direct race
    x_d = np.array([0,1,2,3,4,5,6])
    xticks_d = ['0%','10%', '25%','50%','75%','90%','100%']      
    #subgroup auroc
    auroc_mean_1 = np.array([0.560930326,	0.592787534,	0.630864325,	0.678274219,	0.691697658,	0.683872819,	0.669640725])
    auroc_err_1 = np.array([0.021863406,	0.02086341,	0.01788621,	0.019387452,	0.018874968,	0.017306191,	0.019518777])
    auroc_mean_2 = np.array([0.475448806,	0.568181818,	0.655251951,	0.73431359,	0.742205579,	0.735516529,	0.72706382])
    auroc_err_2 = np.array([0.032680344,	0.038514433,	0.039655859,	0.020465787,	0.015418734,	0.014474826,	0.013127822])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_d)
    ax.set_xticklabels(xticks_d)
    ax.set_xlabel('Disease Prevalence (Black)', fontweight='bold')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    secx = ax.secondary_xaxis('top', functions=(lambda x: 6-x, lambda x: 6-x))
    secx.set_xticklabels(['0%', '0%','10%', '25%','50%','75%','90%','100%'])
    secx.set_xlabel('Disease Prevalence (White)', fontweight='bold')
    ax.plot(x_d, auroc_mean_1, color='slategray', linewidth=3)
    ax.plot(x_d, auroc_mean_2, color='green', linewidth=3)
    ax.fill_between(x_d, auroc_mean_1 - auroc_err_1, auroc_mean_1 + auroc_err_1, facecolor ='slategray', alpha=0.2)
    ax.fill_between(x_d, auroc_mean_2 - auroc_err_2, auroc_mean_2 + auroc_err_2, facecolor ='green', alpha=0.2)
    plt.legend(['Black', 'White'], loc = 'lower right')
    plt.title("Direct Approach-Subgroup AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_direct_race_auroc_subgroup.png"))
    
    #auroc for race classify 
    auroc_mean = np.array([0.839168675,	0.748129448,	0.577669594,	0.690057823,	0.797439882,	0.829075844,	0.848937959])
    auroc_err = np.array([0.015125263,	0.020173565,	0.028751885,	0.020804086,	0.010518939,	0.00929353,	0.009465932]) 
    #auroc for covid classify
    auroc_mean_all = np.array([0.513024564,	0.572680355,	0.64177069,	0.697303002,	0.687000115,	0.672797722,	0.659590077])
    auroc_err_all = np.array([0.01885342,	0.021669654,	0.025473633,	0.014280015,	0.008086937,	0.006436772,	0.007046467])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_d)
    ax.set_xticklabels(xticks_d)
    ax.set_xlabel('Disease Prevalence (Black)', fontweight='bold')
    ax.set_ylabel('AUROC', fontweight='bold')
    plt.ylim(0.4, 1)
    secx = ax.secondary_xaxis('top', functions=(lambda x: 6-x, lambda x: 6-x))
    secx.set_xticklabels(['0%', '0%','10%', '25%','50%','75%','90%','100%'])
    secx.set_xlabel('Disease Prevalence (White)', fontweight='bold')
    ax.plot(x_d, auroc_mean, color='purple', linewidth=3)
    ax.plot(x_d, auroc_mean_all, color='orangered', linewidth=3)
    ax.fill_between(x_d, auroc_mean - auroc_err, auroc_mean + auroc_err, facecolor ='purple', alpha=0.2)
    ax.fill_between(x_d, auroc_mean_all - auroc_err_all, auroc_mean_all + auroc_err_all, facecolor ='orangered', alpha=0.2)
    plt.legend(['Patient Race', 'COVID Status'], title="Measurement of separation", loc = 'lower right')
    plt.title("Direct Approach-Overall AUROC", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_direct_race_auroc_overall.png"))
    
    #sensitivity
    tpr_mean_1 = np.array([0.272424242,	0.367878788,	0.503636364,	0.746515152,	0.895606061,	0.913030303,	0.917424242])
    tpr_err_1 = np.array([0.030868307,	0.024237492,	0.033746861,	0.038720447,	0.02293883,	0.015327264,	0.016144838])
    tpr_mean_2 = np.array([0.737424242,	0.699848485,	0.610454545,	0.536212121,	0.470151515,	0.435757576,	0.402272727])
    tpr_err_2 = np.array([0.026486276,	0.030249321,	0.035325566,	0.033311804,	0.034341113,	0.029614172,	0.020103136])
    
    fig, ax = plt.subplots()
    ax.set_xticks(x_d)
    ax.set_xticklabels(xticks_d)
    ax.set_xlabel('Disease Prevalence (Black)', fontweight='bold')
    ax.set_ylabel('Sensitivity', fontweight='bold')
    plt.ylim(0, 1)
    secx = ax.secondary_xaxis('top', functions=(lambda x: 6-x, lambda x: 6-x))
    secx.set_xticklabels(['0%', '0%','10%', '25%','50%','75%','90%','100%'])
    secx.set_xlabel('Disease Prevalence (White)', fontweight='bold')
    ax.plot(x_d, tpr_mean_1, color='slategray', linewidth=3)
    ax.plot(x_d, tpr_mean_2, color='green', linewidth=3)
    ax.fill_between(x_d, tpr_mean_1 - tpr_err_1, tpr_mean_1 + tpr_err_1, facecolor ='slategray', alpha=0.2)
    ax.fill_between(x_d, tpr_mean_2 - tpr_err_2, tpr_mean_2 + tpr_err_2, facecolor ='green', alpha=0.2)
    plt.legend(['Black', 'White'], loc = 'lower right')
    plt.title("Direct Approach-Subgroup Sensitivity", fontweight='bold')
    plt.savefig(os.path.join(args.save_dir, "spie_direct_race_sensitivity_subgroup.png"))
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',type=str)
    args = parser.parse_args()
    plot_indirect_sex(args)
    plot_indirect_race(args)
    plot_direct_sex(args)
    plot_direct_race(args)
    print("Done\n")