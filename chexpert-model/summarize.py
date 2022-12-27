import os
import pandas as pd
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from constants import *
import itertools
import numpy as np

main_dir = "/gpfs_projects/ravi.samala/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3"

def find_files(args):
    out = {}
    for root,dirs,files in os.walk(main_dir):
        for file in files:
            if file == f"{args.summarize}_summary.csv":
                with open(os.path.join("/".join(root.split("/")[:-1]),"tracking.log"), 'r') as fp:
                    tracking_info = json.load(fp)
                out[os.path.join(root,file)] = tracking_info
    return out

def summarize(args):
    fps = find_files(args)
    for i, fp in enumerate(fps):
        partition_info = fps[fp]['Partition']
        df = pd.read_csv(fp, index_col=0)
        # print(df.head(10))
        print()
        print(fp)
        print("STEP:",int(fp.split("step_")[-1].split("_")[0].replace("/AUROC", '')))
        # return
        # print(df.columns)
        if i == 0:
            cols = ["RAND", "OPTION","Stratify","split","accumulate","step", "dataset"] + df.columns.tolist()
            out_df = pd.DataFrame(columns= cols)
        for ii, row in df.iterrows():

            out_df.loc[len(out_df)] = [partition_info["random_seed"], partition_info['select_option'], partition_info['stratify'], partition_info['split_type'], 
                                       partition_info['accumulate'], int(fp.split("step_")[-1].split("_")[0].replace("/AUROC", '')), ii] + row.values.tolist()
    # return
    out_df.to_csv(os.path.join(main_dir, f'{args.summarize}_overall_summary.csv'), index=False)
    # create shortened version
    # print(out_df.columns)
    for c in cols:
        if "within" in c:
            out_df.pop(c)
        elif "overall" in c:
            out_df.rename(columns = {c:c.replace(" (overall)", "")}, inplace=True)
    out_df.to_csv(os.path.join(main_dir, f'{args.summarize}_overall_summary_short.csv'), index=False)
    # plot(os.path.join(main_dir, f'{args.summarize}_overall_summary.csv'))

def alt_summarize(main_dir=main_dir):
    rcParams['font.family']='monospace'
    rcParams['font.weight']='bold'
    hfont = {'fontname':'monospace', 'fontweight':'bold', 'fontsize':18}
    palette = {"Modality":"tab:red",
          "Sex":"tab:orange",
          "COVID Positive": "tab:green",
          "Race":"tab:blue"}

    # create initial combined df
    in_dfs = []
    acc_splits = {'custom':0, 'custom_acc':1}
    for split in acc_splits:
        for R in range(5):
            for s in range(7):
                if split == 'custom':
                    model_dir = os.path.join(main_dir, f'RAND_{R}_OPTION_0_{split}_7_steps', f'CheXpert_LRcustom3_Epcustom3__step_{s}')
                elif split == 'custom_acc':
                    model_dir = os.path.join(main_dir, f'RAND_{R}_OPTION_0_{split}_7_steps', f'CheXpert_LRcustom4_Epcustom4__step_{s}')
                sum_file = os.path.join(model_dir, "AUROC_summary_last_iter.csv")
                if not os.path.exists(sum_file):
                    continue
                temp_df = pd.read_csv(sum_file)
                temp_df.rename(columns={'Unnamed: 0':'dataset'}, inplace=True)
                temp_df['RAND'] = R
                temp_df['Step'] = s
                temp_df['ACC'] = acc_splits[split]
                
                in_dfs.append(temp_df)
    in_df = pd.concat(in_dfs, ignore_index=True)
    in_df.rename(columns={"CR (overall)":'Modality', 'F (overall)':'Sex', 'Black_or_African_American (overall)':'Race','Yes (overall)':'COVID Positive'}, inplace=True)
    print(in_df)
    print(in_df.dtypes)
    print(in_df.ACC.unique())
    # calculate custom 1.2
    for split in acc_splits:
        for R in range(5):
            for s in range(7):
                if s == 0:
                    continue
                temp_df1 = in_df[(in_df['RAND'] == R) & (in_df['Step'] == s) & (in_df['dataset']=='validation') &(in_df['ACC']==acc_splits[split])]
                temp_df2 = in_df[(in_df['RAND'] == R) & (in_df['Step'] == s-1)& (in_df['dataset']=='next_validation')& (in_df['ACC']==acc_splits[split])]
                if  temp_df1.empty or temp_df2.empty:
                    continue
                # print(R, s)
                # print(temp_df1)
                # print(temp_df2)
                idx = len(in_df)
                in_df.loc[idx] = [0]*len(in_df.columns)
                in_df.at[idx, 'dataset'] = 'custom_1_2'
                in_df.at[idx, 'RAND'] = R
                in_df.at[idx, 'Step'] = s
                in_df.at[idx, 'ACC'] = acc_splits[split]
                for col in ['Modality', 'Sex', 'Race','COVID Positive']:
                    # print()
                    # print(col)
                    current_val = temp_df1[col].values
                    prev_val = temp_df2[col].values
                    # print(current_val, prev_val, current_val-prev_val)
                    in_df.at[idx, col]=current_val-prev_val
    tdf = pd.melt(in_df, id_vars = ['dataset', 'RAND','Step', 'ACC'], var_name='Subgroup')
    tdf.replace({"custom_3_1":"metric 3", 'custom_2_1':"metric 2",'custom_1_2':'metric 1.2', "_":" ",'independent':'Independent'}, regex=True, inplace=True)
    print(tdf['dataset'].unique())
    plot_groups = {
        'Change_Val':['metric 1.2'],
        'Continual_Evaluation_Metrics':['metric 3','metric 2','validation'],
        'Independent_Test_Sets':['Independent test','COVID 19 NY SBU', 'COVID 19 AR', "MIDRC RICORD 1C", 'open RI']
    }
    for key, ds_list in plot_groups.items():
        for task in ['Modality', 'Sex', 'Race','COVID Positive']:
            if key == 'Change_Val':
                continue
            fig, ax = plt.subplots(figsize=(8,6))
            g = sns.relplot(data=tdf[(tdf['dataset'].isin(ds_list)) & (tdf['Subgroup']==task)],
                            kind='line',
                            x='Step',
                            y='value',
                            hue='dataset',
                            col='ACC')
            g.set_axis_labels(x_var='Step', y_var='AUROC', **hfont)
            g.set(ylim=(0.5,1.0))
            plt.title(f"{key.replace('_',' ')} [{task}]", **hfont)
            plt.savefig(os.path.join("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/visuals/", f'LI_{task}_{key}_lineplot.png'), dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            plt.close(fig)
    for ds in tdf['dataset'].unique():
        if ds == 'metric 1.2':
            fig, ax = plt.subplots(figsize=(8,6))
            g = sns.relplot(data=tdf[tdf['dataset'] == ds],
                            kind='line',
                            x='Step',
                            y='value',
                            hue='Subgroup',
                            palette=palette,
                            col='ACC')
            g.set_axis_labels(x_var='Step', y_var='AUROC', **hfont)
            # g.set(ylim=(0.5,1.0))
            plt.title(ds, **hfont)
            plt.savefig(os.path.join("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/visuals/", f'LI_{ds.replace(".","_")}_lineplot.png'), dpi=300, bbox_inches='tight')
            plt.close(g.fig)
            plt.close(fig)
            continue
        fig, ax = plt.subplots(figsize=(8,6))
        g = sns.relplot(data=tdf[tdf['dataset'] == ds],
                        kind='line',
                        x='Step',
                        y='value',
                        hue='Subgroup',
                        palette=palette,
                        col='ACC')
        g.set_axis_labels(x_var='Step', y_var='AUROC', **hfont)
        g.set(ylim=(0.5,1.0))
        plt.title(ds, **hfont)
        plt.savefig(os.path.join("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/visuals/", f'LI_{ds}_lineplot.png'), dpi=300, bbox_inches='tight')
        plt.close(g.fig)
        plt.close(fig)
        
    # temp plotting
    # 
    #
    # in_df.rename(columns={"CR (overall)":'Modality', 'F (overall)':'Sex', 'Black_or_African_American (overall)':'Race','Yes (overall)':'COVID Positive'}, inplace=True)
    # tdf = pd.melt(in_df, id_vars = ['dataset', 'RAND','Step'], var_name='Subgroup')
    # fig, ax = plt.subplots(figsize=(8,6))
    # g = sns.relplot(data=tdf[tdf['dataset']=='independent_test'],
    #                 kind='line',
    #                 x='Step',
    #                 hue='Subgroup',
    #                 y='value')
    # 
    # 
    # plt.savefig("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/temp_independent_lineplot.png",dpi=300,bbox_inches='tight')




def plot(args, summary_dfs=None, summary_csv=None):
    # # TODO: aesthetic options
    fig_size = (8,6)
    ##
    # read in summary_csv, transform as needed
    summary_df = pd.read_csv(summary_csv)
    summary_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    val_col = []
    for col in summary_df.columns:
        if "(" in col:
            if "(overall)" not in col:
                summary_df.pop(col)
                continue
            col_name = col.split(" ")[0]
            summary_df.rename(columns = {col:col_name}, inplace=True)
            val_col.append(col_name)
    label_col = [col for col in summary_df.columns if col not in val_col]
    df = pd.melt(summary_df, id_vars=label_col, value_vars=val_col, var_name = 'task')
    # print(df.head(10))
    # return
    
    bcf_ds = ['validation', 'forward-train','forward-test','backward-test','backward-train']
    bcf_ds = [i for i in bcf_ds if i in df['dataset'].unique()]
    custom_metrics = [i for i in df['dataset'].unique() if 'custom' in i]
    custom_metrics.append('validation')
    compare_ds = [ds for ds in df['dataset'].unique() if ds not in bcf_ds and 'custom' not in ds]
    # compare_ds.append('validation')
    compare_ds.append('joint-validation')
    ds_subsets = {
        'custom':custom_metrics,
        'independent_test_sets':compare_ds
    }
    # first, individual task plots
    
    for sub, tasks in CUSTOM_TASK_SUBSETS.items():
        for key, ds_list in ds_subsets.items():
            p = sns.relplot(data=df[(df['dataset'].isin(ds_list)) & (df['task'].isin(tasks))],
                            x='step',
                            y='value',
                            hue='dataset',
                            kind='line',
                            height=6,
                            aspect=1.3)
            p.set(ylim=(0.5,1.0))
            p.savefig(os.path.join(main_dir, 'visuals',f"{sub}_{key}_lineplot.png"))
            plt.close(p.fig)

    ##  for sub, tasks in CUSTOM_TASK_SUBSETS.items():
    #     # first lineplot - > comparing forward/current/backward
    #     fig, ax = plt.subplots(figsize=fig_size)
    #     g = sns.relplot(data=df[(df['dataset'].isin(bcf_ds)) & (df['task'].isin(tasks))],
    #                 x='step',
    #                 y='value',
    #                 hue='dataset',
    #                 style=None,
    #                 kind='line',
    #                 #errorbar='sd', # TODO: unsure why this doesn't work
    #                 col='task')
    #     g.set(ylim=(0.5,1.0))
    #     plt.savefig(os.path.join(main_dir,'visuals', f"{sub}_bcf_lineplot.png" ), dpi=300)
    #     plt.close()
    #      # second lineplot - > comparing datasets
    #     fig, ax = plt.subplots(figsize=fig_size)
    #     g = sns.relplot(data=df[(df['dataset'].isin(compare_ds)) & (df['task'].isin(tasks))],
    #                 x='step',
    #                 y='value',
    #                 hue='dataset',
    #                 style=None,
    #                 kind='line',
    #                 col='task')
    #     g.set(ylim=(0.5,1.0))
    #     plt.savefig(os.path.join(main_dir,'visuals', f"{sub}_ds_lineplot.png" ), dpi=300)
    #     plt.close()
    #     # third lineplot - comparing custom metrics
    #     fig, ax = plt.subplots(figsize=fig_size)
    #     g = sns.relplot(data=df[(df['dataset'].isin(custom_metrics)) & (df['task'].isin(tasks))],
    #                 x='step',
    #                 y='value',
    #                 hue='dataset',
    #                 style=None,
    #                 kind='line',
    #                 col='task')
    #     g.set(ylim=(0.5,1.0))
    #     plt.savefig(os.path.join(main_dir,'visuals', f"{sub}_custom_metric_lineplot.png" ), dpi=300)
    #     plt.close()
    #     # first boxplot -> comparing forward/current/backward
    #     fig, ax = plt.subplots(figsize=fig_size)
    #     g = sns.FacetGrid(df[(df['dataset'].isin(bcf_ds)) & (df['task'].isin(tasks))], col='dataset', col_order=[*[i for i in bcf_ds if i.startswith('b')], 'validation',*[i for i in bcf_ds if i.startswith('f')]], sharex=False) # add row='split' to compare different splits
    #     g.map_dataframe(sns.boxplot,
    #         x='step',
    #         y='value')
    #     g.map_dataframe(sns.stripplot, 
    #         x='step',
    #         y='value',
    #         color='.25')        
    #     g.set(ylim=(0.5,1.0))
    #     plt.savefig(os.path.join(main_dir,'visuals', f"{sub}_bcf_boxplot.png" ), dpi=300)
    #     plt.close()
    #     # second boxplot -> comparing datasets
    #     fig, ax = plt.subplots(figsize=fig_size)
    #     g = sns.FacetGrid(df[(df['dataset'].isin(compare_ds)) & (df['task'].isin(tasks))], col='dataset')
    #     g.map_dataframe(sns.boxplot,
    #         x='step',
    #         y='value')
    #     g.map_dataframe(sns.stripplot,
    #         x='step',
    #         y='value',
    #         color='.25')
    #     g.set(ylim=(0.5,1.0))
    #     plt.savefig(os.path.join(main_dir,'visuals', f"{sub}_ds_boxplot.png" ), dpi=300)
    #     plt.close()
    #     # third boxplot -> comparing custom metrics
    #     fig, ax = plt.subplots(figsize=fig_size)
    #     g = sns.FacetGrid(df[(df['dataset'].isin(custom_metrics)) & (df['task'].isin(tasks))], col='dataset')
    #     g.map_dataframe(sns.boxplot,
    #         x='step',
    #         y='value')
    #     g.map_dataframe(sns.stripplot,
    #         x='step',
    #         y='value',
    #         color='.25')
    #     g.set(ylim=(0.5,1.0))
    #     plt.savefig(os.path.join(main_dir,'visuals', f"{sub}_custom_metrics_boxplot.png" ), dpi=300)
    #     plt.close()
    # print(df.head(10))
    print("plotting complete")
        
def get_imgs_and_pats(file):
    temp_df = pd.read_csv(file)
    n_imgs = len(temp_df)
    temp_df['patient_id'] = temp_df['Path'].apply(path_to_pid)
    n_pats = len(temp_df['patient_id'].unique())
    # return {"images":n_imgs, 'patients':n_pats}
    return [n_imgs, n_pats]

def path_to_pid(img_path):
    img_name = img_path.split("/")[-1]
    pid = "_".join(img_name.split("_")[:-1])
    return pid 

def partition_info(main_dir=main_dir):
    partition_info = {}
    out_df = pd.DataFrame(columns=['RAND','partition', 'num images', 'num patients'])
    for R in range(5):
        print("R:",R)
        partition_info[R] = {}
        
        rand_dir = os.path.join(main_dir, f'RAND_{R}_OPTION_0_custom_7_steps')
        # partition_info[R]['independent test'] = get_imgs_and_pats(os.path.join(rand_dir,'joint_validation.csv'))
        out_df.loc[len(out_df)] = [R, 'independent test'] + get_imgs_and_pats(os.path.join(rand_dir,'joint_validation.csv'))
        for s in range(7):
            partition_info[R][f"step {s} training"] = get_imgs_and_pats(os.path.join(rand_dir, f"step_{s}.csv"))
            partition_info[R][f"step {s} validation"] = get_imgs_and_pats(os.path.join(rand_dir, f"step_{s}_validation.csv"))
            out_df.loc[len(out_df)] = [R, f"step {s} training"] + get_imgs_and_pats(os.path.join(rand_dir, f"step_{s}.csv"))
            out_df.loc[len(out_df)] = [R, f"step {s} validation"] + get_imgs_and_pats(os.path.join(rand_dir, f"step_{s}_validation.csv"))
        # print(out_df)
        # return
    out_df.to_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/partition_summary.csv", index=False)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--summarize",default='AUROC')
    args = parser.parse_args()
    # summarize(args)

    # plot(args, summary_csv=os.path.join(main_dir, f'{args.summarize}_overall_summary.csv'))

    alt_summarize()
    # partition_info()
    print("Done\n")
