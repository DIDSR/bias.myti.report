import os 
import pandas as pd
import argparse
from datetime import date
import json

# STRATIFICATION VARIABLES ========
# # currently only supports stratification by sex, race, COVID_positive, and/or modality
equal_stratification_groups = ['M-White-Yes-CR', 'F-White-Yes-CR','M-Black-Yes-CR', 'F-Black-Yes-CR',
                               'M-White-No-CR', 'F-White-No-CR','M-Black-No-CR', 'F-Black-No-CR']
custom_composition ={ # to exclude a group from sratification, set all values to 0
    'sex':{"M":1, "F":1},
    'race':{"White":1, "Black":1},
    'COVID_positive':{"Yes":1, "No":1},
    'modality':{"CR":1, "DX":0}
}
# # partition composition options:
    # 'custom' - use the composition specified in custom_composition
    # 'equal' - equal stratification of the subgroups listed in equal_stratification_groups
    # None - no stratification used for this partition
    # Note: due to required rounding at different times, using a custom split results in slight changes to testing and validation sizes
        # (ex. 0.20 may change to 0.194)
train_composition = 'custom'
validation_composition = 'custom'
validation_2_composition = 'equal'
test_composition = 'equal'

# CONSTANTS ===============
subtract_from_smallest_subgroup = 5
group_dict = { 
    'sex':{'subgroups':['M','F'],"loc":['patient_info', 'sex']},
    'race':{'subgroups':['White','Black'],'loc':['patient_info', 'race']},
    'COVID_positive':{'subgroups':['Yes','No'],'loc':['patient_info', 'COVID_positive']},
    'modality':{'subgroups':['CR','DX'], 'loc':['images_info','modality']}
}

conversion_files_openhpc = {repo:f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/{repo}_jpegs/conversion_table.json" for repo in ["MIDRC_RICORD_1C", "COVID_19_NY_SBU", "COVID_19_AR", "open_RI"]}
conversion_files_openhpc['open_A1'] = "/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_open_A1_jpegs/conversion_table.json"

conversion_files_betsy = {repo:f"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/{repo}_jpegs/conversion_table.json" for repo in ["MIDRC_RICORD_1C", "COVID_19_NY_SBU", "COVID_19_AR", "open_RI"]}
conversion_files_betsy['open_A1'] = "/scratch/alexis.burgon/2022_CXR/data_summarization/20221010/open_A1_jpegs/conversion_table.json" 

def bootstrapping(args):
    print("Beginning bootstrapping")
    # 0) check for issues with input arguments ======================================================================
    if args.test_rand is not None:
        if args.test_rand == 'None':
            args.test_rand = None
        elif type(args.test_rand) is not int:
            args.test_rand = int(args.test_rand)
    if args.val_2_rand is not None:
        if args.val_2_rand == 'None':
            args.val_2_rand = None
        elif type(args.val_2_rand) is not int:
            args.val_2_rand = int(args.val_2_rand)
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
    if 'gpfs_projects' in save_folder:
        conversion_table_files = conversion_files_openhpc
    elif 'scratch' in save_folder:
        conversion_table_files = conversion_files_betsy
    # 2) a) create overall dataframe (switch from json formatting to csv) ===========================================
    conversion_tables = pd.concat([pd.read_json(fp) for fp in conversion_table_files.values()])
    input_summaries = []
    for in_summ in args.input_list:
        input_summaries.append(pd.read_json(in_summ, orient='table'))
    df = convert_from_summary(pd.concat(input_summaries, axis=0), conversion_tables, args.min_img_per_patient, args.max_img_per_patient, args.patient_img_selection_mode, args.random_seed + args.random_seed_initial)
    # get a version of the df with all images/patient
    all_df = convert_from_summary(pd.concat(input_summaries, axis=0), conversion_tables, 0, None, 'random', args.random_seed + args.random_seed_initial)
    # 2) b) limit to labels in group_dict (remove others or not depending on arguments passed) ======================
    for grp in group_dict:
        if len(df[df[grp].isin(args.tasks)]) == 0: # not interested in this group as an output task, don't restrict
            continue
        df[grp] = df[grp].replace({t:'other' for t in df[grp].unique() if t not in args.tasks}, regex=True)
        if args.allow_other == "False":
            df = df[df[grp] != 'other']
    img_df = df.copy()
    bp_df = df.drop('Path', axis=1).drop_duplicates() # by-patient df for splitting/stratifying
    # 3) train/validation/test split ================================================================================
    # # process into easier format
    split_list = ['train', 'validation', 'validation_2', 'independent_test']
    splits = pd.DataFrame(columns=['size', 'comp', 'limit_img', 'rand_seed', 'limit_pid', 'get_remaining'],
    index=['train','validation','validation_2','independent_test'])
    splits.at['validation',:] = [args.validation_size, validation_composition, False, None, False, False]
    splits.at['validation_2',:] = [args.validation_size_2, validation_2_composition,False,args.val_2_rand, False, False]
    splits.at['independent_test',:] = [args.test_size, test_composition,False, args.test_rand, False, args.remaining_to_test]
    splits.at['train',:] = [1-splits['size'].sum(), train_composition, False, None, False, False]
    for s in splits.index:
        if s in args.img_splits:
            splits.at[s,"limit_img"] = True
        if s in args.limit_samples:
            splits.at[s, "limit_pid"] = True
    bp_split_dfs = {}
    # 3) a) load any existing partitions
    for s in splits.index:
        if splits.at[s,'rand_seed'] is not None and os.path.exists(os.path.join(save_folder, f"{s}.csv")):
            print(f"Loading {s} split from file...")
            bp_split_dfs[s] = pd.read_csv(os.path.join(save_folder, f"{s}.csv")).drop("Path",axis=1).drop_duplicates()
    # 3) b) check if any partitions limit total number of patients, if so, get that number
    if splits['limit_pid'].any():
        limit_pid_total_num = adjust_comp(bp_df, 'equal', args.random_seed + args.random_seed_initial).groupby('subgroup')['patient_id'].count().min()*int(args.min_num_subgroups)
    # 3) c) get remaining partitions
    for s in splits.index:
        remaining_bp_df = prevent_data_leakage(bp_df, bp_split_dfs.values())
        if s in bp_split_dfs:
            continue
        if splits.at[s,'rand_seed'] is None:
            random_seed = args.random_seed_initial + args.random_seed
        else:
            random_seed = args.random_seed_initial + splits.at[s, 'rand_seed']
        if args.stratify == 'False':
            num = round(splits.at[s,'size']*len(bp_df))
            bp_split_dfs[s] = remaining_bp_df.sample(n=num, random_state=random_seed)
        elif splits.at[s, 'limit_pid']:
            split_number = splits.at[s, 'size'] * limit_pid_total_num
            bp_split_dfs[s] = adjust_comp(remaining_bp_df, splits.at[s,'comp'], random_seed, split_num=split_number)
        else:
            # get new split fraction based on what splits have already been taken
            split_fraction = splits.at[s,'size'] / splits[~splits.index.isin(bp_split_dfs)]['size'].sum()
            bp_split_dfs[s] = adjust_comp(remaining_bp_df, splits.at[s,'comp'], random_seed, split_frac=split_fraction)
    # 4) convert from by-patient dataframes back to by-image dataframes =============================================
    # TODO: more than 1 step
    if not os.path.exists(os.path.join(save_folder, f"RAND_{args.random_seed}")):
        os.mkdir(os.path.join(save_folder, f"RAND_{args.random_seed}"))
    if args.steps == 1: # Saving
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
                    output_files[s].to_csv(os.path.join(save_folder, f"{s}.csv"), index=False)
            else:
                output_files[s].to_csv(os.path.join(save_folder,f"RAND_{args.random_seed}", f"{s}.csv"), index=False)
        bp_summary, img_summary = get_stats(output_files)
        bp_summary.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'by_patient_split_summary.csv'))
        img_summary.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'by_image_split_summary.csv'))
    else:
        print("Not yet implemented for more than one step")
    # 5) save arguments and settings for future reference
    tracking_info = args.__dict__
    tracking_info['training comp'] = train_composition
    tracking_info['validation comp'] = validation_composition
    tracking_info['validation 2 comp'] = validation_2_composition
    tracking_info['testing comp'] = test_composition
    tracking_info['custom comp'] = custom_composition
    tracking_info['equal split subgroups'] = equal_stratification_groups
    tracking_info['Generated on'] = str(date.today())
    with open(os.path.join(save_folder, f"RAND_{args.random_seed}", "partition_info.log"), 'w') as fp:
        json.dump(tracking_info, fp, indent=4)

def adjust_comp(in_df, comp, random_seed, split_frac=None, split_num=None):
    df = in_df.copy()
    if comp is None:
        return df
    elif comp == 'equal':
        sub_dfs = []
        for sub in equal_stratification_groups:
            temp_df = df.copy()
            for s in sub.split("-"):
                for grp, vals in group_dict.items():
                    if s in vals['subgroups']:
                        temp_df = temp_df[temp_df[grp] == s]
            temp_df['subgroup'] = sub
            sub_dfs.append(temp_df)
        subgroup_df = pd.concat(sub_dfs, axis=0)
        n_each_group = subgroup_df.groupby('subgroup')['patient_id'].count().min() - subtract_from_smallest_subgroup
        if n_each_group < 1:
            raise Exception(f"cannot stratify by these subgroups, smallest subgroup has fewer than {subtract_from_smallest_subgroup} patients")
        for ii, sub in enumerate(sub_dfs):
            sub_dfs[ii] = sub.sample(n=n_each_group, random_state=random_seed)
        out_df = pd.concat(sub_dfs, axis=0)
        isub_portions = {x:1 for x in equal_stratification_groups}
    elif comp == 'custom': # TODO: adjust to optimize number of patients in each subgroup, currently sets them equal and then adjusts to composition (except groups not being used)
        sub_dict = {}
        for grp in custom_composition:
            grp_total = sum(custom_composition[grp].values())
            grp_max = max(custom_composition[grp].values())
            if grp_total == 0: # Not stratifying by this group
                continue
            if 'subgroup' not in df.columns:
                df['subgroup'] = df[grp]
            else:
                df.loc[:,'subgroup'] = df['subgroup'] + "-" + df[grp]
            df = df[df[grp].isin(custom_composition[grp])]
            for subgrp in custom_composition[grp]:
                sub_dict[subgrp] = (custom_composition[grp][subgrp] / grp_max)
                # remove subgroups that we aren't using
                if sub_dict[subgrp] == 0:
                    df = df[df[grp]!=subgrp]
        # set remaining subgroups equal -> adjust to fit comp
        n_each_group = df.groupby('subgroup')['patient_id'].count().min() - subtract_from_smallest_subgroup
        if n_each_group < 1:
            raise Exception(f"cannot stratify by these subgroups, smallest subgroup has fewer than {subtract_from_smallest_subgroup} patients")
        sub_dfs = []
        isub_portions = {}
        for sub in df['subgroup'].unique():
            isub_portions[sub] = 1
            for s in sub.split("-"):
                isub_portions[sub] *= sub_dict[s]
            sub_dfs.append(df[df['subgroup'] == sub].sample(n=round(n_each_group*isub_portions[sub]), random_state=random_seed))
        out_df = pd.concat(sub_dfs, axis=0)
    if split_frac is None and split_num is None:
        return out_df
    elif split_num is not None:
        sub_dfs = []
        for sub in out_df['subgroup'].unique():
            sub_dfs.append(out_df[out_df['subgroup'] == sub].sample(n=round(split_num*(isub_portions[sub]/sum(isub_portions.values()))), random_state=random_seed))
        return pd.concat(sub_dfs, axis=0)
    else:
        sub_dfs = []
        for sub in out_df['subgroup'].unique():
            sub_dfs.append(out_df[out_df['subgroup'] == sub].sample(frac=split_frac, random_state=random_seed))
        return pd.concat(sub_dfs, axis=0)

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
        df[x[1]] = df.apply(lambda row: row[x[0]][0][x[1]], axis=1)
    df['race'] = df['race'].replace({" ":"_"},regex=True)
    df['race'] = df['race'].replace({"Black_or_African_American":"Black"}, regex=True)
    # df = df.explode('images')
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
    # for grp in group_dict:
    #     if 'subgroup' not in df.columns:
    #         df['subgroup'] = df[grp]
    #     else:
    #         df.loc[:,'subgroup'] = df['subgroup'] + "-" + df[grp]
    return df.rename(columns = {'patient_id':'number of patients'}).groupby(['split','subgroup'])['number of patients'].nunique(), df.rename(columns={'patient_id':'number of images'}).groupby(['split','subgroup'])['number of images'].count()
    

def old_get_stats(test_df, val_df, val_2_df, train_df):
    test_df= test_df.copy()
    val_df = val_df.copy()
    val_2_df = val_2_df.copy()
    train_df = train_df.copy()
    test_df.loc[:,'split'] = 'test'
    val_df.loc[:,'split'] = 'validation'
    
    train_df.loc[:,'split'] = 'train'
    if len(val_2_df) != 0:
        val_2_df.loc[:,'split'] = 'validation_2'
        df = pd.concat([test_df, val_df, train_df, val_2_df], axis=0)
    else:
        df = pd.concat([test_df, val_df, train_df], axis=0)
    for sp in ['test', 'validation','validation_2', 'train']: # check that there is no patient_id_overlap
        if df[df['split']!=sp]['patient_id'].isin(df[df['split']==sp]['patient_id']).any():
            print(f"OOPS: {sp}")
    for grp in group_dict:
        if 'subgroup' not in df.columns:
            df['subgroup'] = df[grp]
        else:
            df.loc[:,'subgroup'] = df['subgroup'] + "-" + df[grp]
    return df.rename(columns = {'patient_id':'number of patients'}).groupby(['split','subgroup'])['number of patients'].nunique(), df.rename(columns={'patient_id':'number of images'}).groupby(['split','subgroup'])['number of images'].count()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # # partition general settings (input file, classes, split sizes, stratification)
    parser.add_argument("-i","--input_list", action='append', required=True, default=[], help="list of input summary files")
    parser.add_argument("-tasks", action='append', default=[], required=True)
    parser.add_argument("-stratify", type=str, default="False", help='if True, will follow the train/valid/test compositions at the top of \
        this file. If False, will randomly split train/test/validation')
    parser.add_argument("-allow_other", type=str, default="True", help='if False, restricts the patients that can be used to the subgroups listed in group_dict')
    parser.add_argument("-test_size", type=float, required=True)
    parser.add_argument("-validation_size", type=float, required=True)
    parser.add_argument("-validation_size_2", type=float, default=0.0)
    parser.add_argument("-consistent_test_random_state", default=None, dest='test_rand',
                        help="""If None, each random seed will have it's own independent test set,
                            if an integer is passed, that will be used for the random seed for the
                            single independent test partition""")
    parser.add_argument("-consistent_validation_2_random_state", default=None, dest='val_2_rand')
    parser.add_argument("-remaining_to_test", default=False)
    # # settings for multiple steps [WIP]
    parser.add_argument("-steps", default=1)
    # # reproducibility 
    parser.add_argument("-random_seed_initial", type=int, default=2022)
    parser.add_argument("-random_seed", default=1)
    # # saving/naming
    parser.add_argument("-partition_name", type=str, required=True)
    parser.add_argument("-save_dir", type=str, required=True)
    # # number of images per patient
    parser.add_argument("-img_splits", default=[], action='append', help="the splits that the img selection settings will be applied to")
    parser.add_argument("-min_img_per_patient", default=0)
    parser.add_argument("-max_img_per_patient", default=None)
    parser.add_argument("-patient_img_selection_mode", default='random', choices=['random', 'first','last'])
    # # limiting overall number of patients
    parser.add_argument("-limit_samples", default=[], action='append')
    parser.add_argument("-min_num_subgroups", default=None)
    
    bootstrapping(parser.parse_args())
    print("\nDONE\n")