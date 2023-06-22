import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import os


def summarize(partition_dfs, args):
    mpl.rcParams['axes.titleweight'] = 'bold'
    summ_dir = os.path.join(args.RAND_dir, "partition_summary")
    if not os.path.exists(summ_dir):
        os.mkdir(summ_dir)
    summarize_distributions(partition_dfs, args, summ_dir)
    check_partitions(partition_dfs, args, summ_dir)
    if args.steps > 1:
        for p in args.partitions:
            check_partitions(partition_dfs, args, summ_dir, partition=p)

def summarize_distributions(partition_dfs, args, summ_dir):
    output_format = "svg" # TODO: allow change
    stat = 'percent'
    colors = sns.color_palette().as_hex() # TODO: allow custom color palette
    atts = list(set(args.attributes + args.summary_attributes))
    for a in atts + ['subgroup']:
        if args.partition_summary_type == 'bar':
            mainfig = plt.figure(figsize=( args.steps*6,(len(args.partitions)+1)*2), layout='constrained')
        elif args.partition_summary_type == 'pie':
            mainfig = plt.figure(figsize=( args.steps*6,(len(args.partitions)+1)*3), layout='constrained')
        subfigs = mainfig.subfigures(1, args.steps, squeeze=False)
        for s in range(args.steps):
            axes = subfigs[0][s].subplots(len(args.partitions)+1, 1)
            if args.steps > 1:
                subfigs[0][s].suptitle(f"Step {s}")
            for ii, p in enumerate(['Overall'] + args.partitions):
                if p == 'Overall':
                    df = pd.concat( partition_dfs[f"Step {s}"].values(), axis=0)
                    c=colors[1]
                else:
                    df = partition_dfs[f"Step {s}"][p]
                    c=colors[0]
                df = df[[args.id_col, a]].copy().drop_duplicates()
                if len(df) != df[args.id_col].nunique():
                    print(f"Trouble summarizing {a} by {args.id_col} ({len(df)} v. {df[args.id_col].nunique()}) -- Summary figures may be inaccurate for this attribute!")
                axes[ii].set_title(f"{p} ({df[args.id_col].nunique()} {args.id_col}s)")
                if args.partition_summary_type == 'bar':
                    sns.histplot(data=df, x=a, ax=axes[ii], color=c, stat=stat)
                    if stat == 'percent':
                        axes[ii].set_ylim(0,100)
                    elif stat == 'portion':
                        axes[ii].set_ylim(0,1)
                elif args.partition_summary_type == 'pie':
                    df = df.groupby(a)[args.id_col].nunique().reset_index()
                    axes[ii].pie(df[args.id_col], labels=df[a], colors=colors, autopct="%.1f%%")
        plt.savefig(os.path.join(summ_dir, f"Distribution_summary__{a}.{output_format}"))
        plt.close("all")
    # get summary csv
    out_info = pd.DataFrame(columns=['Step', 'Partition', 'Subgroup', 'Num Patients', 'Num Images'])
    for s in range(args.steps):
        for p in args.partitions:
            for sub in partition_dfs[f"Step {s}"][p].subgroup.unique():
                temp_df = partition_dfs[f"Step {s}"][p][partition_dfs[f"Step {s}"][p]['subgroup'] == sub]
                out_info.loc[len(out_info)] = [s, p, sub, temp_df[args.id_col].nunique(), len(temp_df)]
    if args.steps == 1:
        out_info = out_info.drop('Step', axis=1)
    out_info.to_csv(os.path.join(summ_dir, "Partition_summary.csv"), index=False)


def check_partitions(indiv_dfs,  args, save_dir, partition='all', by_id=True, output_format="svg"):
    """ Checks + plots partition overlap """
    if partition == 'all':
        partitions = args.partitions
    else:
        partitions=[partition]
    idx = []
    
    for p in partitions:
        for s in range(args.steps):
            idx.append(f"Step {s} - {p}")
    df = pd.DataFrame(0,index=idx, columns=idx)
    for p1 in idx:
        for p2 in idx:
            df1 = indiv_dfs[p1.split(" - ")[0]][p1.split(" - ")[1]]
            df2 = indiv_dfs[p2.split(" - ")[0]][p2.split(" - ")[1]]
            if by_id:
                df1 = df1[args.id_col].unique()
                df2 = df2[args.id_col].unique()
            else:
                df1 = df1["Path"].unqiue()
                df2 = df2["Path"].unique()
            df.at[p1,p2] = len(set(df1)&set(df2))
    sns.heatmap(df, square=True, annot=True, fmt=".0f")
    plt.title("Partition Overlap")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Partition_overlap__{partition}.{output_format}", dpi=300)
    plt.close('all')
    

            
    



            