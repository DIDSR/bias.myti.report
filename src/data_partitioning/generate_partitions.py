from constants import *
from argument_parser import PartitionArgumentParser
from utils import *
from subgroup_distribution_options import distributions
import pandas as pd
import sys
import os

def generate_partitions(args):
    # file management TODO: move (part of?) to argument parser
    if len(args.batch) != 0 and os.path.exists(args.batch_dir):
        existing_csvs = [os.path.join(args.batch_dir, f) for f in os.listdir(args.batch_dir) if f.endswith(".csv")]
    else:
        existing_csvs = []

    if not os.path.exists(args.save_loc):
        os.mkdir(args.save_loc)
    if len(args.batch) != 0 and not os.path.exists(args.batch_dir):
        os.mkdir(args.batch_dir)
    if not os.path.exists(args.RAND_dir):
        os.mkdir(args.RAND_dir)
    elif not args.overwrite:
        raise Exception(f"A directory already exists at {args.RAND_dir}. Use --overwrite to overwrite.")
    
    partition_dfs = load_existing_csvs(existing_csvs, args)
    repo_dfs = []
    for repo in args.repos:
        summary_fp, conversion_fp = get_repository_files(repository=repo, betsy=args.betsy)
        portable_fp = get_portable_file(repository=repo, betsy=args.betsy)
        df = convert_summary_format(summary_fp, conversion_fp, attributes=args.attributes, repo=repo, portable_fp=portable_fp)
        test_date_fp = get_test_date_file(repository=repo, betsy=args.betsy)
        df = select_patient_images(df, args, test_date_csv=test_date_fp)
        if not args.allow_not_reported: # remove samples that are missing information associated with 1+ of args.attributes
            for a in args.attributes:
                df = df[df[a] != 'Not Reported']
        repo_dfs.append(df)
    df = pd.concat(repo_dfs, axis=0, ignore_index=True)
    
    # remove excluded attributes
    for i, j in args.exclude_attributes.items():
        if i not in df.columns:
            raise Exception(f"Cannot exclude based on attribute {i}, not in available attributes: {df.columns}")
        df = df[~df[i].isin(j)].copy()
    columns = ['Path', args.id_col, 'subgroup'] + list(set(args.attributes + args.summary_attributes))
    df = df.drop([c for c in df.columns if c not in columns], axis=1) # remove unecessary columns
    step_dfs = determine_distributions(df, args, mode='step', existing_dfs=partition_dfs) 
    for s in range(args.steps):
        partition_dfs[f"Step {s}"] = determine_distributions(step_dfs[f"Step {s}"], args, mode='partition', existing_dfs = partition_dfs[f"Step {s}"], step=s)
    if len(args.accumulate) > 0:
        for i, a in enumerate(args.accumulate):
            if a == "True":
                partition_dfs = accumulate(partition_dfs, partition=args.partitions[i], steps=args.steps)
    if len(args.replace) > 0:
        for i, r in enumerate(args.replace):
            if r == 'True':
                partition_dfs = replace(partition_dfs, partition=args.partitions[i], steps=args.steps, RAND=args.RAND, id_col=args.id_col, by_subgroup=args.replace_by_subgroup)
    summarize(partition_dfs, args)
    for s in range(args.steps):
        for p in partition_dfs[f"Step {s}"]:
            if args.steps == 1:
                if p in args.batch:
                    partition_out_file = os.path.join(args.batch_dir, f"{p}.csv")
                    if os.path.exists(partition_out_file) and not args.overwrite:
                        continue
                else:
                    partition_out_file = os.path.join(args.RAND_dir, f"{p}.csv")
            else:
                if p in args.batch:
                    partition_out_file = os.path.join(args.batch_dir, f"step_{s}__{p}.csv")
                    if os.path.exists(partition_out_file) and not args.overwrite:
                        continue
                else:
                    partition_out_file = os.path.join(args.RAND_dir, f"step_{s}__{p}.csv")
            if len(args.tasks) > 0:# create task columns
                temp_df = pd.get_dummies(partition_dfs[f"Step {s}"][p][["Path"] + args.tasks].set_index("Path"), prefix='', prefix_sep='')
                partition_dfs[f"Step {s}"][p] = pd.merge(partition_dfs[f"Step {s}"][p], temp_df, on="Path")
            print(f"Saving the {p} partition (step {s}) to {partition_out_file}")
            partition_dfs[f"Step {s}"][p].to_csv(partition_out_file, index=False)            

if __name__ == "__main__":
    parser = PartitionArgumentParser()
    args = parser.parse_args()
    generate_partitions(args)
    log_partition_settings(args)