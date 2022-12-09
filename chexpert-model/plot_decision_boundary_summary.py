import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import numpy as np

db_folder = 'decision_boundaries_all'

input_classes = {'sex':['M','F'], 'race':['White', 'Black_or_African_American'], "COVID_positive":["Yes", "No"]}
output_classes = ['Yes', "No"]
plot_classes = ['Yes']
abbreviation_table = {
    'Female':"F",
    'Male':"M",
    'CR':"C",
    "DX":"D",
    "White":"W",
    'Black_or_African_American':"B",
    "Yes":"P",# positive
    "No":'N'
}
# model_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/1_step_all_CR_stratified_ind_test/RAND_0/CHEXPERT_RESNET_0__step_0/"
model_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/1_step_all_CR_stratified_ind_test_BETSY/RAND_19/CHEXPERT_RESNET_0__step_0/"

def plot_db_results(model_dir=model_dir,
                    input_classes=input_classes,
                    output_classes=output_classes,
                    plot_classes=plot_classes):
    summary_file = os.path.join(model_dir, db_folder,'overall_summary.csv')
    df = pd.read_csv(summary_file, index_col=0)
    for i, j in abbreviation_table.items():
        df = df.replace(to_replace=i, value=j, regex=True)
    # get the list of triplet classes
    id_cols = [col for col in df.columns if col in input_classes]
    # return
    df['triplet'] = None
    
    for ii, row in df.iterrows():
        df.at[ii,'triplet'] = "".join(row[id_cols].values)
    for task in plot_classes:
        df[f"%{task} difference"] = abs(df[f'%{task} expected'] -df[f'%{task} (mean)'])
    # read the individual summary files
    triplet_classes = df['triplet'].values
    triplet_summaries = []
    input_combos = combinations(input_classes.keys(), 2)
    for t in triplet_classes:
        t_summ_fp = os.path.join(model_dir, db_folder, f'{t}_{t}_{t}_summary.csv')
        if not os.path.exists(t_summ_fp):
            continue
        t_df = pd.read_csv(t_summ_fp, index_col=0)
        t_df['triplet'] = t
        for key in input_classes:
            t_df[key] = df[df['triplet'] == t][key].values[0]
        triplet_summaries.append(t_df)
    trip_df = pd.concat(triplet_summaries, axis=0, ignore_index=True)
    for task in plot_classes:
        trip_df[f"%{task} difference"] = 0.0
        # get the difference from the expected value
        for ii, row in trip_df.iterrows():
            expected_value = df[df['triplet'] == row['triplet']][f'%{task} expected'].values
            trip_df.at[ii, f"%{task} difference"] = abs(expected_value - row[f"%{task}"] )[0]
        # set color by class -> keep consistent between runs
        color_dict = {
            "FWN":"#4878D0",
            "FWP":"#EE854A",
            "FBN":"#6ACC64",
            "FBP":"#D65F5F",
            "MWN":"#956CB4",
            "MWP":"#8C613C",
            "MBN":"#DC7EC0",
            "MBP":"#797979"
        }
        # # BOX PLOT ==============================
        fig,ax = plt.subplots(figsize=(8,6))
        sorted_df = df.sort_values(by=f'%{task} difference')
        sns.boxplot(data=trip_df, x='triplet', y=f'%{task} difference', order=sorted_df['triplet'],showmeans=True,
                    meanprops={"marker":"o", 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':10}, palette=color_dict)
        plt.title("percent difference from expected")
        ax.set_ylim(0,100)
        plt.savefig(os.path.join(model_dir, db_folder,f'{task}_difference_boxplot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # # BAR PLOT ===================================
        fig,ax = plt.subplots(figsize=(8,6))
        sns.barplot(data=trip_df, x='triplet', y=f'%{task} difference', order=sorted_df['triplet'], palette=color_dict)
        plt.title("percent difference from expected")
        ax.set_ylim(0,75)
        plt.savefig(os.path.join(model_dir, db_folder,f'{task}_difference_barplot.png'),dpi=300, bbox_inches='tight')
        plt.close()
        # # HEAT MAPS =============
        for interaction_group in input_combos:
            fig,ax = plt.subplots(figsize=(8,6))
            sns.heatmap(data=trip_df.pivot_table(index=interaction_group[0], columns=interaction_group[1], values=f'%{task} difference', aggfunc=np.mean),
                        vmin=10, vmax=50, annot=True, cmap='viridis')
            plt.title("mean percent difference from expected")
            plt.savefig(os.path.join(model_dir, db_folder,f'{task}_difference_{interaction_group[0]}_{interaction_group[1]}_heatmap.png'),dpi=300, bbox_inches='tight')
            plt.close()
            # return
        
def compare_n_samples(n_samples=[50,100,150,200,250,300,350,400,450,495], random_state=0):
    summary_file = os.path.join(model_dir, 'decision_boundaries','overall_summary.csv')
    df = pd.read_csv(summary_file, index_col=0)
    for i, j in abbreviation_table.items():
        df = df.replace(to_replace=i, value=j, regex=True)
    # get the list of triplet classes
    id_cols = [col for col in df.columns if col in input_classes]
    # return
    df['triplet'] = None
    
    for ii, row in df.iterrows():
        df.at[ii,'triplet'] = "".join(row[id_cols].values)
    for task in plot_classes:
        df[f"%{task} difference"] = abs(df[f'%{task} expected'] -df[f'%{task} (mean)'])
    # read the individual summary files
    triplet_classes = df['triplet'].values
    triplet_summaries = []
    input_combos = combinations(input_classes.keys(), 2)
    for t in triplet_classes:
        t_summ_fp = os.path.join(model_dir, db_folder, f'{t}_{t}_{t}_summary.csv')
        if not os.path.exists(t_summ_fp):
            continue
        t_df = pd.read_csv(t_summ_fp, index_col=0)
        t_df['triplet'] = t
        for key in input_classes:
            t_df[key] = df[df['triplet'] == t][key].values[0]
        triplet_summaries.append(t_df)
    trip_df = pd.concat(triplet_summaries, axis=0, ignore_index=True)
    for task in plot_classes:
        trip_df[f"%{task} difference"] = 0.0
        # get the difference from the expected value
        for ii, row in trip_df.iterrows():
            expected_value = df[df['triplet'] == row['triplet']][f'%{task} expected'].values
            trip_df.at[ii, f"%{task} difference"] = abs(expected_value - row[f"%{task}"] )[0]
        # set color by class -> keep consistent between runs
        color_dict = {
            "FWN":"#4878D0",
            "FWP":"#EE854A",
            "FBN":"#6ACC64",
            "FBP":"#D65F5F",
            "MWN":"#956CB4",
            "MWP":"#8C613C",
            "MBN":"#DC7EC0",
            "MBP":"#797979"
        }
        fig, ax = plt.subplots(figsize=(20,6), nrows=2, ncols=5, sharey=True)
        overall_sort = df.sort_values(by=f'%{task} difference').index
        order_df = pd.DataFrame(index=[1,2,3,4,5,6,7,8])

        for ii, n in enumerate(n_samples):
            # randomly sample the correct number from each triplet group
            temp_df = trip_df.groupby('triplet').sample(n=n, random_state=random_state)
            sorted_df = temp_df.groupby('triplet')[f'%{task} difference'].mean()
            sorted_df = sorted_df.sort_values()
            if ii < 5:
                c = ii
            else:
                c = ii - 5
            order_df[n] = sorted_df.index
            sns.barplot(data=temp_df, x='triplet', y=f"%{task} difference", order=sorted_df.index, palette=color_dict, ax=ax[int(ii/5),int(c)])
            # ax[int(ii/5),c].set_xticks([])
            # ax[int(ii/5),c].set_xticklabels(sorted_df.index,rotation=45)
            ax[int(ii/5),c].set_xlabel("")
            ax[int(ii/5),c].set_ylabel("")
            ax[int(ii/5),c].set_title(f"{n} samples/triplet")
            
        
        # plt.legend(labels=df.triplet.unique())
        print(order_df)
        order_df.to_csv(os.path.join(model_dir, db_folder, f"num_samples_{random_state}.csv"))
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, db_folder, f"num_samples_{random_state}.png"))
        plt.close()

if __name__ == '__main__':
    plot_db_results()
    # compare_n_samples()
    print("\nDone\n")
