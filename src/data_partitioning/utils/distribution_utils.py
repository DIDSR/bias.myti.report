from subgroup_distribution_options import distributions, attribute_abbreviations
from itertools import product
import pandas as pd
import numpy as np

def determine_distributions(df, args, mode, existing_dfs, step=None): # TODO: load existing partitions using existing_dfs
    """ Determine the number of patients/subgroup in each step """
    # print(f"Partitioning {mode}")
    # Get by-patient df
    image_df = df.copy()
    cols = list(set(args.attributes + [args.id_col, 'subgroup']))
    df = image_df[cols].copy().drop_duplicates()
    # if args.id_col != 'Path':
    #     df = df.drop(columns=['Path']).drop_duplicates()
    if len(df) != df[args.id_col].nunique() and not 'BraTS' in args.repos:
        print(len(df), df[args.id_col].nunique())
        raise Exception("Issue converting by-image df to by-id") 
    attribute_options = {att:df[att].unique().tolist() for att in args.attributes}
    possible_subgroups = list(product(*attribute_options.values()))
    if mode == 'step':
        subgroup_df = pd.DataFrame(columns=[f"Step {s}" for s in range(args.steps)], index=["-".join(x) for x in possible_subgroups]).rename_axis("Subgroup").rename_axis('Step',axis=1)
        dists = args.step_distributions
        sizes = args.step_sizes
        if len(args.batch) > 0:
            RANDs = [args.batch_RAND+args.RAND for s in range(args.steps)]
        else:
            RANDs = [args.RAND for s in range(args.steps)]
        step_dfs = {} # switch partition dfs to step dfs
        for s in existing_dfs:
            if len(existing_dfs[s]) > 0:
                step_dfs[s] = pd.concat(existing_dfs[s].values(), axis=0)
        existing_dfs = step_dfs
    elif mode == 'partition':
        subgroup_df = pd.DataFrame(columns=[p for p in args.partitions], index=["-".join(x) for x in possible_subgroups]).rename_axis("Subgroup").rename_axis('Partition',axis=1)
        dists = args.partition_distributions
        sizes = args.partition_sizes
        if len(args.constant_partitions) > 0 and step != 0:
            for i, p in enumerate(args.partitions):
                if p in args.constant_partitions:
                    sizes[i] = 0
        sizes = [s/sum(sizes) for s in sizes]
        RANDs = [args.partition_RANDs[p] for p in args.partitions]
    cols = subgroup_df.columns
    subgroup_counts = df.groupby('subgroup')[args.id_col].nunique()
    subgroup_counts -= args.subtract_from_smallest
    overall_total = df[args.id_col].nunique()
    if len(existing_dfs) > 0: # remove samples already used
        used_samples = pd.concat(existing_dfs.values(), axis=0)
        df = df[~df[args.id_col].isin(used_samples[args.id_col])]
    indiv_dfs = {}
    if 'random' in dists:
        if len(set(dists)) == 1: # all partitions are random
            for i, c in enumerate(cols):
                total_size = int(sizes[i]*overall_total)
                if c in existing_dfs:
                    total_size -= existing_dfs[c][args.id_col].nunique()
                temp_df = df.sample(n=int(total_size), random_state=RANDs[i])
                # add the samples already existing
                if c in existing_dfs:
                    temp_df = pd.concat([temp_df, existing_dfs[c]], axis=0)
                indiv_dfs[c] = image_df[image_df[args.id_col].isin(temp_df[args.id_col])].copy()
                df = df[~df[args.id_col].isin(indiv_dfs[c][args.id_col])] # prevent ids from being used in multiple steps/partitions  
        else:
            raise NotImplementedError()
    else:
        for i, c in enumerate(cols):
            if dists[i] == 'random':
                raise NotImplementedError()
            else:
                dist = get_distribution(dists[i], args.attributes, possible_subgroups)
                subgroup_df[c] = subgroup_df.index.map(dist['portion'])
                subgroup_df[c] *= sizes[i]
        subgroup_df['total'] = subgroup_df.sum(axis=1)
        subgroup_df /= subgroup_df['total'].sum()
        rel_subgroup_counts = subgroup_counts.divide(subgroup_df['total']) 
        rel_subgroup_counts.replace([np.inf, -np.inf], np.nan, inplace=True)
        subgroup_df = subgroup_df * rel_subgroup_counts.min()
        print(f"Limiting subgroup for partitioning is {rel_subgroup_counts.idxmin()}")
        subgroup_df.replace(np.nan, 0, inplace=True)
        for x in existing_dfs:
            num_ids_already = existing_dfs[x].groupby('subgroup')[args.id_col].nunique()
            for subgroup, num in num_ids_already.items():
                subgroup_df.at[subgroup, x] -= num
        # print(subgroup_df)
        for i, c in enumerate(cols):
            temp_dfs = []
            if c in existing_dfs:
                temp_dfs.append(existing_dfs[c])
                
            for ii, row in subgroup_df.iterrows():
                if int(row[c]) > 0:
                    temp_dfs.append(df[df['subgroup'] == ii].sample(n=int(row[c]), random_state=args.RAND))
            
            indiv_dfs[c] = image_df[image_df[args.id_col].isin(pd.concat(temp_dfs, axis=0)[args.id_col])].copy()
            df = df[~df[args.id_col].isin(indiv_dfs[c][args.id_col])] # prevent ids from being used in multiple steps/partitions
    return indiv_dfs
    
def abbreviate_subgroups(subgroup):
    attributes = subgroup.split("-")
    abbrev_subgroup = ""
    for a in attributes:
        if a in attribute_abbreviations:
            abbrev_subgroup += attribute_abbreviations[a]
        else:
            abbrev_subgroup += a
    return abbrev_subgroup

def abbreviate_att(attribute):
    if attribute in attribute_abbreviations:
        return attribute_abbreviations[attribute]
    else:
        return attribute

def get_portion(row, dist):
    row = row.apply(abbreviate_att)
    for entry in distributions[dist].keys():
        if set(entry).issubset(list(row.values)):
            return distributions[dist][entry]
    

def get_distribution(dist, attributes, possible_subgroups):
    df = pd.DataFrame(possible_subgroups, columns=attributes)
    # df = df.applymap(abbreviate_att)
    if dist == 'equal':
        df['portion'] = 1
    elif dist in distributions.keys():
        df['portion'] = df.apply(lambda row: get_portion(row, dist), axis=1)
    else:
        raise Exception(f" distribution {dist} not recognized; see subgroup_distribution_options.py")
    df['portion'] /= df['portion'].sum()
    df['subgroup'] = df[attributes].apply("-".join, axis=1)
    return df.set_index('subgroup')


        
