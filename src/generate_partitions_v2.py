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
    'race':{"White":.50, "Black":.50},
    'COVID_positive':{"Yes":1, "No":1},
    'modality':{"CR":1, "DX":0}
}
# # partition composition options:
    # 'custom' - use the composition specified in custom_composition
    # 'equal' - equal stratification of the subgroups listed in equal_stratification_groups
    # None - no stratification used for this partition
    # Note: due to required rounding at different times, using a custom split results in slight changes to testing and validation sizes
        # (ex. 0.20 may change to 0.194)
train_composition = 'equal'
validation_composition = 'equal'
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
    # 3) a) Test split ----------------------------------------------------------------------------------------------
    if args.test_rand is None:
        test_random_seed = args.random_seed + args.random_seed_initial # same seed used for validation/training
    else:
        test_random_seed = args.test_rand + args.random_seed_initial
        
    if os.path.exists(os.path.join(save_folder, 'independent_test.csv')) and args.test_rand is not None:
        print("\narguments indicate a single independent test for all RAND values, and a test csv already exists, loading...")
        test_bp_df = pd.read_csv(os.path.join(save_folder, 'independent_test.csv')).drop("Path", axis=1).drop_duplicates()
        trv_bp_df = bp_df[~bp_df['patient_id'].isin(test_bp_df['patient_id'])]
    else:
        if args.stratify == 'False':
            test_bp_df = bp_df.sample(frac=args.test_size, random_state=test_random_seed)
        else:
            test_bp_df = adjust_comp(bp_df, test_composition, test_random_seed, split_frac=args.test_size)
        # remove the patients used in the test partition
        trv_bp_df = bp_df[~bp_df['patient_id'].isin(test_bp_df['patient_id'])]
    # 3) b) Validation split ----------------------------------------------------------------------------------------
    if args.stratify == 'False':
        val_num = round(args.validation_size*len(bp_df))
        val_bp_df = trv_bp_df.sample(n=val_num, random_state=args.random_seed + args.random_seed_initial)
    else:
        strat_bp_df = adjust_comp(bp_df, validation_composition, args.random_seed + args.random_seed_initial)
        strat_trv_bp_df = adjust_comp(trv_bp_df, validation_composition, args.random_seed + args.random_seed_initial)
        val_n_size = (len(strat_bp_df)/len(strat_trv_bp_df)) * args.validation_size
        val_bp_df = adjust_comp(trv_bp_df, validation_composition, args.random_seed + args.random_seed_initial, split_frac=val_n_size)
    remaining_bp_df = bp_df[~bp_df['patient_id'].isin(val_bp_df['patient_id'])]
    remaining_bp_df = remaining_bp_df[~remaining_bp_df['patient_id'].isin(test_bp_df['patient_id'])]
    # 3) c) Validation part 2 ---------------------------------------------------------------------------------------
    if args.val_2_rand is None:
        val_2_random_seed = args.random_seed + args.random_seed_initial
    else:
        val_2_random_seed = args.val_2_rand + args.random_seed_initial
    if os.path.exists(os.path.join(save_folder, 'validation_2.csv')) and args.val_2_rand is not None:
        print("\narguments indicate a single validation 2 file for all RAND values, and a validation 2 csv already exists, loading...")
        val_2_bp_df = pd.read_csv(os.path.join(save_folder, 'validation_2.csv')).drop("Path", axis=1).drop_duplicates()
    else:
        if args.stratify == 'False':
            val_num_2 = round(args.validation_size_2*len(bp_df))
            val_2_bp_df = trv_bp_df.sample(n=val_num_2, random_state=val_2_random_seed)
        else:
            strat_bp_df = adjust_comp(bp_df, validation_2_composition, val_2_random_seed)
            strat_trv_bp_df = adjust_comp(remaining_bp_df, validation_2_composition, val_2_random_seed)
            val_2_n_size = (len(strat_bp_df)/len(strat_trv_bp_df)) * args.validation_size_2
            val_2_bp_df = adjust_comp(remaining_bp_df, validation_2_composition, val_2_random_seed, split_frac=val_2_n_size)
    
    # 3) d) Train Split ---------------------------------------------------------------------------------------------
    # tr_bp_df = remaining_bp_df[~remaining_bp_df['patient_id'].isin(val_2_bp_df['patient_id'])]
    tr_bp_df = bp_df[~bp_df['patient_id'].isin(test_bp_df['patient_id'])]
    tr_bp_df = tr_bp_df[~tr_bp_df['patient_id'].isin(val_bp_df['patient_id'])]
    tr_bp_df = tr_bp_df[~tr_bp_df['patient_id'].isin(val_2_bp_df['patient_id'])]
    if args.stratify == "False":
        train_bp_df = tr_bp_df.copy()
    else:
        train_bp_df = adjust_comp(tr_bp_df, train_composition, args.random_seed + args.random_seed_initial)
    # 4) convert from by-patient dataframes back to by-image dataframes =============================================
    # TODO: more than 1 step
    if not os.path.exists(os.path.join(save_folder, f"RAND_{args.random_seed}")):
        os.mkdir(os.path.join(save_folder, f"RAND_{args.random_seed}"))
    if args.steps == 1: # Saving
        if args.remaining_to_test:
            test_bp_df = bp_df[~bp_df['patient_id'].isin(train_bp_df['patient_id'])]
            test_bp_df = test_bp_df[~test_bp_df['patient_id'].isin(val_bp_df['patient_id'])]
            test_bp_df = test_bp_df[~test_bp_df['patient_id'].isin(val_2_bp_df['patient_id'])]
        test_df = img_df[img_df['patient_id'].isin(test_bp_df['patient_id'])]
        valid_df = img_df[img_df['patient_id'].isin(val_bp_df['patient_id'])]
        valid_2_df = img_df[img_df['patient_id'].isin(val_2_bp_df['patient_id'])]
        train_df = img_df[img_df['patient_id'].isin(train_bp_df['patient_id'])]
        bp_summary, img_summary = get_stats(test_df, valid_df,valid_2_df, train_df)
        bp_summary.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'by_patient_split_summary.csv'))
        img_summary.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'by_image_split_summary.csv'))
        test_df = convert_to_csv(test_df, args.tasks)
        if args.test_rand is not None and not os.path.exists(os.path.join(save_folder, 'independent_test.csv')): # single independent test for all RAND
            test_df.to_csv(os.path.join(save_folder, 'independent_test.csv'), index=False)
        elif args.test_rand is None:
            test_df.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'independent_test.csv'), index=False)
        else:
            print("\nsingle joint independent test file already exists")
        valid_df = convert_to_csv(valid_df, args.tasks)
        valid_df.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'validation.csv'), index=False)
        valid_2_df = convert_to_csv(valid_2_df, args.tasks)
        if args.val_2_rand is not None and not os.path.exists(os.path.join(save_folder, 'validation_2.csv')): # single validation 2 for all rand
            valid_2_df.to_csv(os.path.join(save_folder, 'validation_2.csv'),index=False)
        elif args.val_2_rand is None:
            valid_2_df.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'validation_2.csv'), index=False)
        else:
            print("\nsingle validation 2 file already exists")
        train_df = convert_to_csv(train_df, args.tasks)
        train_df.to_csv(os.path.join(save_folder, f"RAND_{args.random_seed}", 'train.csv'), index=False)
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

def adjust_comp(in_df, comp, random_seed, split_frac=None):
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
        for sub in df['subgroup'].unique():
            isub_portion = 1
            for s in sub.split("-"):
                isub_portion *= sub_dict[s]
            sub_dfs.append(df[df['subgroup'] == sub].sample(n=round(n_each_group*isub_portion), random_state=random_seed))
        out_df = pd.concat(sub_dfs, axis=0)
    if split_frac is None:
        return out_df
    else:
        sub_dfs = []
        for sub in out_df['subgroup'].unique():
            sub_dfs.append(out_df[out_df['subgroup'] == sub].sample(frac=split_frac, random_state=random_seed))
        return pd.concat(sub_dfs, axis=0)

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
    df = df.explode('images')
    df['Path'] = df['images'].map(conversion_table.set_index('dicom')['jpeg'])
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
    return df
        
def get_stats(test_df, val_df, val_2_df, train_df):
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
    parser.add_argument("-min_img_per_patient", default=0)
    parser.add_argument("-max_img_per_patient", default=None)
    parser.add_argument("-patient_img_selection_mode", default='random', choices=['random', 'first','last'])
    
    bootstrapping(parser.parse_args())
    print("\nDONE\n")