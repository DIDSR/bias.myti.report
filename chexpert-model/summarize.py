import os
from re import L
import pandas as pd
import argparse

main_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/scenario_1"

def find_files(args):
    out = {}
    for root,dirs,files in os.walk(main_dir):
        for file in files:
            if f"{args.summarize}_summary.csv" in file:
                out[os.path.join(root,file)] = {}
                with open(os.path.join("/".join(root.split("/")[:-1]),"log.log"), 'r') as fp:
                   for line in fp:
                    if ":" in line:
                        s_line = line.replace(",","").replace("\n","").replace("\"","").replace(" ","").split(":")
                        out[os.path.join(root,file)][s_line[0]]  = s_line[1]
    return out


def summarize(args):
    fps = find_files(args)
    for i, fp in enumerate(fps):
        df = pd.read_csv(fp, index_col=0)
        if i == 0:
            cols = ["RAND", "OPTION","split","accumulate","step", "dataset"] + df.columns.tolist()
            out_df = pd.DataFrame(columns= cols)
        for ii, row in df.iterrows():
            out_df.loc[len(out_df)] = [fps[fp]["random_seed"], fps[fp]['select_option'], fps[fp]['split_type'], fps[fp]['accumulate'], fp.split("step_")[-1].split("_")[0], ii] + row.values.tolist()
    
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
