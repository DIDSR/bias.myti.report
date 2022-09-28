import os
import pandas as pd
import argparse
import json

main_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/scenario_1"

def find_files(args):
    out = {}
    for root,dirs,files in os.walk(main_dir):
        for file in files:
            if f"{args.summarize}_summary.csv" in file:
                with open(os.path.join("/".join(root.split("/")[:-1]),"tracking.log"), 'r') as fp:
                    tracking_info = json.load(fp)
                out[os.path.join(root,file)] = tracking_info
    return out

def summarize(args):
    fps = find_files(args)
    for i, fp in enumerate(fps):
        partition_info = fps[fp]['Partition']
        df = pd.read_csv(fp, index_col=0)
        if i == 0:
            cols = ["RAND", "OPTION","Stratify","split","accumulate","step", "dataset"] + df.columns.tolist()
            out_df = pd.DataFrame(columns= cols)
        for ii, row in df.iterrows():
            out_df.loc[len(out_df)] = [partition_info["random_seed"], partition_info['select_option'], partition_info['stratify'], partition_info['split_type'], 
                                       partition_info['accumulate'], fp.split("step_")[-1].split("_")[0], ii] + row.values.tolist()
    
    out_df.to_csv(os.path.join(main_dir, f'{args.summarize}_overall_summary.csv'), index=False)
    # create shortened version
    for c in cols:
        if "within" in c:
            out_df.pop(c)
        elif "overall" in c:
            out_df.rename(columns = {c:c.replace(" (overall)", "")}, inplace=True)
    out_df.to_csv(os.path.join(main_dir, f'{args.summarize}_overall_summary_short.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s","--summarize",default='AUROC')
    args = parser.parse_args()
    summarize(args)
    print("Done\n")
