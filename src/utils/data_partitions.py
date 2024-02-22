import os 
import pandas as pd
import argparse
from datetime import date
import json

# STRATIFICATION VARIABLES ========
# # currently only supports stratification by sex, race, COVID_positive, and/or modality

# for open_A1
equal_stratification_groups = ['M-White-Yes-CR', 'F-White-Yes-CR','M-Black-Yes-CR', 'F-Black-Yes-CR',
                               'M-White-No-CR', 'F-White-No-CR','M-Black-No-CR', 'F-Black-No-CR']
# CONSTANTS ===============
subtract_from_smallest_subgroup = 5
group_dict = { 
    'sex':{'subgroups':['M','F'],"loc":['patient_info', 'sex']},
    'race':{'subgroups':['White','Black'],'loc':['patient_info', 'race']},
    'COVID_positive':{'subgroups':['Yes','No'],'loc':['patient_info', 'COVID_positive']},
    'modality':{'subgroups':['CR','DX'], 'loc':['images_info','modality']}
}

def bootstrapping(args):
    """ 
    Partitioning patient data into train, validation, validation_2 and testing.
    In each dataset, patient can be partitioned equally or by customized ratio in race, sex, image modality and COVID. 
    
    """
    print("Beginning bootstrapping")
    # 0) check for issues with input arguments ======================================================================
    if type(args.random_seed) is not int:
        args.random_seed = int(args.random_seed)
    args.min_img_per_patient = int(args.min_img_per_patient)
    if args.max_img_per_patient is not None and type(args.max_img_per_patient) != int:
        if args.max_img_per_patient == 'None':
            args.max_img_per_patient = None
        else:
            args.max_img_per_patient = int(args.max_img_per_patient)
    if args.remaining_to_test == 'False':
        args.remaining_to_test = False
    elif args.remaining_to_test == 'True':
        args.remaining_to_test = True
    # 1) set up save location, summary files
    save_folder = os.path.join(args.save_dir, args.partition_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    conversion_table_files = "/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_open_A1_jpegs/conversion_table.json"
    tasks = ["M" "F" 'White' 'Black' 'Yes' 'No']
    # 2) a) create overall dataframe (switch from json formatting to csv) ===========================================
    conversion_tables = pd.concat([pd.read_json(fp) for fp in conversion_table_files.values()])
    input_summaries = []
    for in_summ in args.input_list:
        input_summaries.append(pd.read_json(in_summ, orient='table'))
    df = convert_from_summary(pd.concat(input_summaries, axis=0), conversion_tables, args.min_img_per_patient, args.max_img_per_patient, args.patient_img_selection_mode, args.random_seed)
    df = adjust_subgroups(df)
    # get a version of the df with all images/patient
    all_df = convert_from_summary(pd.concat(input_summaries, axis=0), conversion_tables, 0, None, 'random', args.random_seed)
    all_df = adjust_subgroups(all_df)
    # 2) b) limit to labels in group_dict (remove others or not depending on arguments passed) ======================
    for grp in group_dict:
        if len(df[df[grp].isin(tasks)]) == 0: # not interested in this group as an output task, don't restrict
            continue
        df[grp] = df[grp].replace({t:'other' for t in df[grp].unique() if t not in tasks}, regex=True)
        df = df[df[grp] != 'other']
    bp_df = df.drop('Path', axis=1).drop_duplicates() # by-patient df for splitting/stratifying
    print("\nNumber of patients/subgroup in input summary:")
    print(bp_df[bp_df['subgroup'].isin(equal_stratification_groups)].groupby("subgroup")['patient_id'].count())
    # 3) train/validation/test split ================================================================================
    # # process into easier format
    splits = pd.DataFrame(columns=['size', 'limit_img', 'rand_seed', 'get_remaining'],
    index=['train','validation','independent_test'])
    splits.loc['validation',:] = [args.validation_size, False, None, False]
    splits.loc['independent_test',:] = [args.test_size, False, None, args.remaining_to_test]
    splits.loc['train',:] = [1-splits['size'].sum(), False, None, False]
    for s in splits.index:
        if args.max_img_per_patient is not None:
            splits.at[s,"limit_img"] = True
    bp_split_dfs = {}
    # get remaining partitions
    for s in splits.index:
        remaining_bp_df = prevent_data_leakage(bp_df, bp_split_dfs.values())
        if s in bp_split_dfs:
            continue
        random_seed = args.random_seed
        # get new split fraction based on what splits have already been taken
        split_fraction = splits.at[s,'size'] / splits[~splits.index.isin(bp_split_dfs)]['size'].sum()
        bp_split_dfs[s] = adjust_comp(remaining_bp_df, random_seed, split_frac=split_fraction)
    # 4) convert from by-patient dataframes back to by-image dataframes =============================================  
    if not os.path.exists(os.path.join(save_folder, f"RAND_{args.random_seed}")):
        os.mkdir(os.path.join(save_folder, f"RAND_{args.random_seed}"))

    output_files = {}
    for s in splits.index:
        if splits.at[s,'get_remaining']: # this is the split that gets samples that do not belong to one of the equal_stratification_groups
            temp_df = all_df[~all_df['subgroup'].isin(equal_stratification_groups)]
            bp_split_dfs[s] = pd.concat([bp_split_dfs[s], temp_df])
        if splits.at[s, 'limit_img']:
            output_files[s] = df[df['patient_id'].isin(bp_split_dfs[s]['patient_id'])]
        else:
            output_files[s] = all_df[all_df['patient_id'].isin(bp_split_dfs[s]['patient_id'])]
        # save output files
        if splits.at[s, 'rand_seed'] is not None:
            if os.path.exists(os.path.join(save_folder, f"{s}.csv")):
                print(f"Joint {s} file already exists, not overwriting")
            else:
                temp_df = convert_to_csv(output_files[s], args.tasks)
                temp_df.to_csv(os.path.join(save_folder, f"{s}.csv"), index=False)
        else:
            temp_df = convert_to_csv(output_files[s], args.tasks)
            temp_df.to_csv(os.path.join(save_folder,f"RAND_{args.random_seed}", f"{s}.csv"), index=False)
    bp_summary, img_summary = get_stats(output_files)
    bp_summary.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'by_patient_split_summary.csv'))
    img_summary.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'by_image_split_summary.csv'))

    # 5) save arguments and settings for future reference
    tracking_info = args.__dict__
    tracking_info['equal split subgroups'] = equal_stratification_groups
    tracking_info['Generated on'] = str(date.today())
    with open(os.path.join(save_folder, f"RAND_{args.random_seed}", "partition_info.log"), 'w') as fp:
        json.dump(tracking_info, fp, indent=4)

def adjust_subgroups(in_df):
    ''' adjust subgroup information displayed in dataframe to only reflect attributes that are specified as import in equal_stratification_groups'''
    df = in_df.copy()
    rel_subs = list(set([x for y in equal_stratification_groups for x in y.split("-")]))
    irrel_groups = [x for x in group_dict if len([z for z in group_dict[x]['subgroups'] if z in rel_subs]) == 0]
    # remove irrelevant groups from the subgroup labels
    for x in irrel_groups:
        df['subgroup'] = df.apply(lambda row: row['subgroup'].replace(row[x], ""), axis=1)
    # # remove extraneous -'s
    df['subgroup'] = df['subgroup'].replace("--","-", regex=True)
    idx = df[df['subgroup'].str.endswith("-")].index
    df.loc[idx, 'subgroup'] = df.loc[idx, 'subgroup'].apply(lambda x: x[:-1])
    idx = df[df['subgroup'].str.startswith("-")].index
    df.loc[idx, 'subgroup'] = df.loc[idx, 'subgroup'].apply(lambda x: x[1:])
    return df

def adjust_comp(in_df, random_seed, split_frac=1, split_num=None):
    '''
        adjusts the composition of in_df to match the comp specified, if split_frac or split_num are specified,
    generates a split of the specified size.
    -------
    in_df - input dataframe (by-patient)
    random_seed - random control
    split_frac - the fraction (decimal) of the available data to return
    split_num - the number of patients overall to return
    '''
    df = in_df.copy()
    subgroup_proportions = {sub:1 for sub in equal_stratification_groups}
    # joint process for both custom and equal comp type ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    # convert from subgroup proportions to specific percentage of the split
    subgroup_percentages = {i:(j/sum(subgroup_proportions.values())) for i,j in subgroup_proportions.items()}
    # get values for each subgroup
    if split_num is None: # figure out actual split num based on limiting group
        # determine limiting group
        # below method only works if all compositions = custom, otherwise it messes stuff up
        max_size_by_subgroup = {}
        for sub in subgroup_percentages:
            original_n_sub = df[df['subgroup'] == sub]['patient_id'].count()
            max_size_by_subgroup[sub] = original_n_sub / subgroup_percentages[sub]
        split_num = int(min(max_size_by_subgroup.values()))
    sub_dfs = []
    for sub in subgroup_percentages:
        sub_num = round(split_num * subgroup_percentages[sub])
        sub_dfs.append(df[df['subgroup'] == sub].sample(n=round(sub_num*split_frac), random_state=random_seed))
    adjusted_df = pd.concat(sub_dfs, axis=0)
    return adjusted_df

def prevent_data_leakage(base_df, df_list:list):
    out_df = base_df.copy()
    for df in df_list:
        out_df = out_df[~out_df['patient_id'].isin(df['patient_id'])]
    return out_df

def convert_to_csv(df, tasks):
    for grp in group_dict:
        temp_df = df[grp].str.get_dummies()
        df = pd.concat([df, temp_df], axis=1)
    # add columns for tasks that aren't in the df
    for t in tasks:
        if t not in df.columns:
            df[t] = 0
    cols = ['patient_id', 'Path'] + tasks
    return df[cols]

def convert_from_summary(df, conversion_table, min_img, max_img, selection_mode, random_state):
    for val in group_dict.values():
        x = val['loc']
        df = df[df['images_info'].str.len() != 0].copy() # remove rows without image information
        df[x[1]] = df.apply(lambda row: row[x[0]][0][x[1]], axis=1)
    df['race'] = df['race'].replace({" ":"_"},regex=True)
    df['race'] = df['race'].replace({"Black_or_African_American":"Black"}, regex=True)
    # to get the study date with the image, we need to joint-explode images and images_information
    exp_cols = {'images', 'images_info'}
    other_cols = list(set(df.columns)-set(exp_cols))
    exploded = [df[col].explode() for col in exp_cols]
    temp_df = pd.DataFrame(dict(zip(exp_cols, exploded)))
    temp_df = df[other_cols].merge(temp_df, how='right', left_index=True, right_index=True)
    temp_df = temp_df.loc[:, df.columns] # get original column order
    df = temp_df.copy()
    df['Path'] = df['images'].map(conversion_table.set_index('dicom')['jpeg'])
    # get study date
    df['study date'] = df['images_info'].apply(lambda x: x['study date'])
    df = df.sort_values(['study date'])
    # remove patients with fewer than min_img images
    df = df[df['num_images']>min_img]
    if max_img is not None: # select images from patients with more than max_img imgs
        gb = df[df['num_images']>max_img].groupby('patient_id')
        if selection_mode == 'random':
            group_list = [data.sample(n=max_img, random_state=random_state) for _, data in gb]
        elif selection_mode == 'first':
            group_list = [data.iloc[:max_img] for _, data in gb]
        elif selection_mode == 'last':
            group_list = [data.iloc[-max_img:] for _, data in gb]
        group_list.append(df[df['num_images']<=max_img])
        df = pd.concat(group_list, axis=0)
    cols = ['patient_id','Path']+[grp for grp in group_dict]
    rm_cols = [col for col in df.columns if col not in cols]
    df = df.drop(rm_cols, axis=1)
    df['subgroup'] = df.apply(lambda row: get_subgroup(row), axis=1)
    return df

def get_subgroup(row):
    subgroup = ""
    for g in group_dict:
        if len(subgroup) == 0:
            subgroup = row[g]
        else:
            subgroup = subgroup + "-" + row[g]
    return subgroup

def get_stats(df_dict):
    df_list = []
    for id, split_df in df_dict.items():
        temp_df = split_df.copy()
        temp_df['split'] = id
        df_list.append(temp_df)
    df = pd.concat(df_list, axis=0)
    for sp in df['split'].unique():
        if df[df['split']!=sp]['patient_id'].isin(df[df['split']==sp]['patient_id']).any():
            print(f"OOPS: {sp}")
    return pd.pivot_table(df,values='patient_id', index='subgroup', columns='split',aggfunc=pd.Series.nunique, margins=True), pd.pivot_table(df,values='patient_id', index='subgroup', columns='split',aggfunc='count')
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # # partition general settings (input file, classes, split sizes, stratification)
    parser.add_argument("--input_list", action='append', required=True, default=[], help="list of input summary files")
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--validation_size", type=float, required=True)
    parser.add_argument("--remaining_to_test", default=False)
    parser.add_argument("--random_seed", default=0)
    # # saving/naming
    parser.add_argument("--partition_name", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    # # number of images per patient
    parser.add_argument("--img_splits", default=[], action='append', help="the splits that the img selection settings will be applied to")
    parser.add_argument("--min_img_per_patient", default=0)
    parser.add_argument("--max_img_per_patient", default=None)
    parser.add_argument("--patient_img_selection_mode", default='random', choices=['random', 'first','last'])    
    bootstrapping(parser.parse_args())
    print("\nDONE\n")