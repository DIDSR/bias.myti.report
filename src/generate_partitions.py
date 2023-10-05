'''
    Program that partitions data based on the input summary json files
'''
from cmath import e
from decimal import MIN_EMIN
import os
import argparse
import json
import datetime
import functools
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from datetime import date
import numpy as np
import itertools
import time


subtract_from_smallest_subgroup = 5
RAND_SEED_INITIAL=2022

subgroup_dict = {
    "F,M":["patient_info","sex"],
    "CR,DX":["images_info","modality"],
    "Yes,No":["patient_info","COVID_positive"],
    "Black or African American,White":["patient_info","race"]
}
conversion_files_openhpc = {repo:f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/{repo}_jpegs/conversion_table.json" for repo in ["MIDRC_RICORD_1C", "COVID_19_NY_SBU", "COVID_19_AR", "open_RI"]}
conversion_files_openhpc['open_A1'] ="/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_open_A1_jpegs/conversion_table.json"

# conversion_files_betsy = {repo:f"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/{repo}_jpegs/conversion_table.json" for repo in ["MIDRC_RICORD_1C", "COVID_19_NY_SBU", "COVID_19_AR", "open_AI", "open_RI"]}
conversion_files_betsy = {repo:f"/scratch/alexis.burgon/2022_CXR/data_summarization/20221010/{repo}_jpegs/conversion_table.json" for repo in ["open_A1"]}

def select_image_per_patient(patient_df, n_images):
    # # iterate by patient, sort by "study date" and select the first n_images
    for index, each_patient in patient_df.iterrows():
        num_images = len(each_patient['images_info'])
        patient_study_dates = [a["study date"] for a in each_patient['images_info']]
        date_seq = [datetime.datetime.strptime(ts, "%Y%m%d") for ts in patient_study_dates]
        sorted_date_index = [sorted(date_seq).index(x) + 1 for x in date_seq]
        selected_images_index = sorted_date_index[0:min(n_images, num_images)]
        # # update the patient-level info using the selected index of images
        each_patient['images_info'] = [each_patient['images_info'][i-1] for i in selected_images_index]
        each_patient['images'] = [each_patient['images'][i-1] for i in selected_images_index]
        patient_df.loc[index, 'images_info'] = each_patient['images_info']
        patient_df.loc[index, 'images'] = each_patient['images']
    return patient_df

def get_n_patients(args, df):
    # adjust dataframe columns
    for key, val in subgroup_dict.items():
        df[val[1]] = df.apply(lambda row: row[val[0]][0][val[1]], axis=1)
        df = df[df[val[1]].isin(key.split(','))]
    if args.select_option == 1:
        select_idxs = []
        for index, each_patient in df.iterrows():
            num_images = len(each_patient['images_info'])
            if num_images < args.min_num_image_per_patient:
                continue
            select_idxs.append(index)
        df = df.loc[select_idxs]
        df = select_image_per_patient(df, args.min_num_image_per_patient)

    if args.stratify != "False":
        # get all combinations of the groups specified
        # args.tasks = args.tasks.replace("_", " ")
        args.stratify = args.stratify.split(",")
        subclasses = [key.split(",") for key, item in subgroup_dict.items() if item[1] in args.stratify]
        strat_groups = ["-".join(item) for item in list(itertools.product(*subclasses))]
        strat_idxs = {strat:None for strat in strat_groups}
        strat_dfs = {strat:None for strat in strat_groups}
        for strat in strat_groups:
            sub_idxs = {}
            for sub in strat.split("-"):
                for key, val in subgroup_dict.items():
                    if sub not in key.split(","):
                        continue
                    sub_idxs[sub] = df[df[val[1]] == sub].index.tolist()
                strat_idxs[strat] = find_overlap(sub_idxs)
                strat_dfs[strat] = df.loc[strat_idxs[strat]]
        min_subgroup_size = min([len(item) for item in strat_idxs.values()]) - subtract_from_smallest_subgroup
        if min_subgroup_size <= 0:
            raise Exception(f"Minimum subgroup size is {min_subgroup_size}, cannot stratify")
        # print(min_subgroup_size)
        new_df = pd.concat([strat_dfs[strat].sample(n=min_subgroup_size, random_state=RAND_SEED_INITIAL+args.random_seed) for strat in strat_idxs])
        return new_df
    else:
        return  df      

def bootstrapping(args):
    if "," in args.split_type: # catch problems with custom split sizes
        args.split_type = [float(x) for x in args.split_type.split(",")]
        if len(args.split_type) != args.steps:
            print(f"Given {len(args.split_type)} split sizes for {args.steps} splits")
            return
        elif sum(args.split_type) != 1:
            print(f"{args.split_type} does not sum to 1")
            return
    # import inputs from json files
    dfs = []
    repo_str = ''
    for each_summ_file in args.input_list:
        dfs += [pd.read_json(each_summ_file, orient='table')]
        repo_str += '__' + os.path.basename(each_summ_file).split('.')[0]
    df = functools.reduce(lambda left,right: pd.concat([left, right], axis=0),dfs)
    df = df.reset_index()
    df = df[df.num_images > 0]  # # drop rows with 0 images
    df = df[df['images_info'].map(lambda d: len(d)>0)] ## drop rows with no image information
    total_patients = len(df)
    print(f"\nFound {total_patients} patients from input file(s)")
    print(f"Splitting into {args.steps} steps")
    print("Accumulate: ", args.accumulate)
    print("Stratify: ", args.stratify)
    print("Select Option:", args.select_option)
    print("Split Type:", args.split_type, "\n")
    args.tasks = args.tasks.split(",")
    # load conversion files
    conversion_tables = {}
    if "gpfs_projects" in args.output_dir:
        conversion_files = conversion_files_openhpc
    elif "scratch" in args.output_dir:
        conversion_files = conversion_files_betsy
    else:
        print("could not find conversion files")
    for c, fp in conversion_files.items():
        if not os.path.exists(fp): # DEBUG
            print(f"No file at {fp}")
            return
        conversion_tables[c] = pd.read_json(fp)
    # deal with any settings that can reduce the number of possible samples available
    new_df = get_n_patients(args, df)
    
    if args.stratify != "False":
        # new_df, min_subgroup_size, strat_idxs = get_n_patients(args,df)
        tr_sample, val_split = train_test_split(new_df, test_size=args.percent_test_partition, random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True, stratify=new_df[args.stratify])
        
    else:
        # new_df = get_n_patients(args, df)
        tr_sample, val_split = train_test_split(new_df, test_size=args.percent_test_partition, random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True)
    # separate + save validation split (uses all patients not used in the training sample, not caring for stratification/min image)
    # tr_idxs = tr_sample.index.tolist()
    # val_idxs =[idx for idx in df.index.tolist() if idx not in tr_idxs]
    # val_split = df.loc[val_idxs]
    # validation bootstrapping:
    # TODO: allow unequal steps (?)
    if args.add_joint_validation != "True":
        val_n = int(len(val_split)/args.steps)
    else:
        val_n = int(len(val_split)/(args.steps+1))
    # print(val_n)
    # debug
    # print("Overall val split:")
    # print(val_split['sex'].value_counts())
    for n in range(args.steps):
        print(f"Generating val {n}...")
        
        if args.stratify != "False": # currently set to work with single stratify choice (ex. sex) that has only two subgroups
            step_val = val_split.groupby(args.stratify, group_keys=False).apply(lambda x: x.sample(n=int(val_n/2), random_state=RAND_SEED_INITIAL+args.random_seed))
            # debug
            # print(f"\nStep {n} validation")
            # print(step_val['sex'].value_counts())
            # print("\nRemaining overall")
            # print(val_split['sex'].value_counts())
            
        else:
            step_val = val_split.sample(n=val_n, random_state=RAND_SEED_INITIAL+args.random_seed)
        val_split.drop(step_val.index, axis=0, inplace=True)
        step_val.to_json(os.path.join(args.output_dir, f"step_{n}_validation.json"), orient='table', indent=2)
        step_csv = convert_to_csv(args, step_val, conversion_tables)
        step_csv.to_csv(os.path.join(args.output_dir, f"step_{n}_validation.csv"))
        # print(step_val['race'].value_counts())
        # return # debug
        # print(f"step {n} val:", len(step_val))
    

    if args.add_joint_validation == "True":
        # creates a testing parition from the leftover samples in val_split
        val_split.to_json(os.path.join(args.output_dir, "joint_validation.json"), orient='table', indent=2)
        
        val_csv = convert_to_csv(args, val_split, conversion_tables)
        
        val_csv.to_csv(os.path.join(args.output_dir,"joint_validation.csv"))
        # print("\nJoint val:")
        # print(val_split['sex'].value_counts())
        # print("joint val:", len(val_split))
    
    step_sizes = get_split_sizes(args, len(tr_sample))
    # print(step_sizes)
    step_dfs = {}
    # print(f"Overall tr")
    # print(tr_sample['sex'].value_counts())
    # separate individual steps
    for n in range(args.steps):
        print(f"Generating tr {n}")
        if n == args.steps-1:
            if args.stratify != "False": # currently set to work with single stratify choice (ex. sex) that has only two subgroups
                step_dfs[n] = tr_sample.groupby(args.stratify, group_keys=False).sample(n=int(step_sizes[n]/2), random_state=RAND_SEED_INITIAL+args.random_seed)
            else:
                step_dfs[n] = tr_sample.sample(n=step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed)
        else:
            if args.stratify != "False":
                _, step_dfs[n] = train_test_split(tr_sample, test_size = step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True, stratify=tr_sample[args.stratify])
            else:
                _, step_dfs[n] = train_test_split(tr_sample, test_size = step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True)
            # step_dfs[n] = tr_sample.sample(n=step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed)
            tr_sample.drop(step_dfs[n].index, axis=0, inplace=True)
            # print(f"step {n}")
            # print(step_dfs[n]['sex'].value_counts())
        if args.accumulate != "False" and n > 0:
            if args.accumulate == "True": #complete accumulation
                step_dfs[n] = pd.concat([step_dfs[n-1], step_dfs[n]])
            else: # partial accumulation
                args.accumulate = float(args.accumulate)
                n_acc = round(args.accumulate*len(step_dfs[n-1]))
                acc_samples = step_dfs[n-1].sample(n=n_acc, random_state=RAND_SEED_INITIAL+args.random_seed)
                step_dfs[n] = pd.concat([step_dfs[n],acc_samples])
        # save
        step_dfs[n].to_json(os.path.join(args.output_dir, f"step_{n}.json"), orient='table', indent=2)
        step_csv = convert_to_csv(args, step_dfs[n], conversion_tables)
        step_csv.to_csv(os.path.join(args.output_dir, f"step_{n}.csv"))
        print("step ", n, len(step_csv), "images")

def convert_to_csv(args, df, conversion_tables):
    # converts dataframes to work as csv input for the model (by image rather than by patient)
    csv_df = pd.DataFrame(columns=["patient_id","Path"]+[task for task in args.tasks])
    for iii, row in df.iterrows():
        if type(row['images']) == str:
            row['images'] = [row['images']]
            row['images_info'] = [row['images_info']]
        for ii, img in enumerate(row['images']):
            if ii >= len(row['images_info']): # there are a couple images that are missing information? (open_RI)
                continue
            img_info = {}
            # get jpeg path
            if row['repo'] == "RICORD-1c":
                conv_table = conversion_tables['MIDRC_RICORD_1C']
            else:
                conv_table = conversion_tables[row['repo']]
            if len(conv_table[conv_table['dicom']==img]['jpeg'].values) == 0:
                print(img)
                continue
            img_info['Path'] = conv_table[conv_table['dicom']==img]['jpeg'].values[0]
            img_info['patient_id'] = row['patient_id']
            for task in args.tasks:
                for key, val in subgroup_dict.items():
                    if task.replace("_"," ") in key.split(','):
                        tval = val
                if 'patient_info' in tval:
                    img_info[task] = int(row[tval[0]][0][tval[1]] == task.replace("_"," "))
                else:
                    
                    img_info[task] = int(row[tval[0]][ii][tval[1]] == task.replace("_", " "))
            csv_df.loc[len(csv_df)] = img_info
    return csv_df

def get_split_sizes(args):
    split_sizes = [0 for i in range(args.steps)]
    if args.split_type == 'equal':
        for i in range(args.steps):
            split_sizes[i] = (1/args.steps)
    elif args.split_type == 'increasing':
        tot_splits = (args.steps*args.steps + args.steps)/2
        for i in range(args.steps):
            split_sizes[i] = (1/tot_splits)*(i+1)
    elif args.split_type == 'random':
        random.seed(RAND_SEED_INITIAL+args.random_seed)
        x = random.sample(range(1,10), args.steps)
        y = [z/sum(x) for z in x]
        for i in range(args.steps):
            split_sizes[i] =y[i]
    elif type(args.split_type) == list:
        for i in range(args.steps):
            split_sizes[i] = args.split_type[i]
    else:
        print("unrecognized split_type")
        return
    return split_sizes

def find_overlap(sub_idxs):
    ''' Find the overlap between sub group idxs'''
    out = list(sub_idxs.values())[0]
    for i in range(1, len(sub_idxs)):
        out = [value for value in out if value in list(sub_idxs.values())[i]]
    return out       

def convert_to_csv_v2(args, df, conversion_table):
    for x in subgroup_dict.values():
        df[x[1]] = df.apply(lambda row: row[x[0]][0][x[1]], axis=1)
    df['race'] = df['race'].replace({" ":"_"}, regex=True)
    new_df = df.explode('images')
    new_df['Path'] = new_df['images'].map(lambda i: conversion_table[conversion_table['dicom']==i]['jpeg'].values[0])
    # drop no longer needed columns
    new_df = new_df.drop(['patient_info','images_info','num_images', 'bad_images', 'bad_images_info', 'index', 'images'], axis=1)
    return new_df.reset_index()

def bootstrapping_v2(args):
    print('Beginning partition generation')
    args.tasks = args.tasks.split(",")
    if "," in args.split_type: # catch problems with custom split sizes
        args.split_type = [float(x) for x in args.split_type.split(",")]
        if len(args.split_type) != args.steps:
            print(f"Given {len(args.split_type)} split sizes for {args.steps} splits")
            return
        elif sum(args.split_type) != 1:
            print(f"{args.split_type} does not sum to 1")
            return
    # 1) Set up overall csv file
        # Includes all samples to be used in all random seeds
    overall_csv = os.path.join("/".join(args.output_dir.split("/")[:-1]), 'all_partition_samples.csv')
    # print(overall_csv)
    if os.path.exists(overall_csv): # csv already exists, read from file
        print("overall csv found, reading from file")
        df = pd.read_csv(overall_csv, index_col=0)
    else: # generate and save overall csv
        print("no overall csv found, generating")
        
        # load json files
        dfs = []
        repo_str = ''
        for each_summ_file in args.input_list:
            dfs.append(pd.read_json(each_summ_file, orient='table'))
            repo_str = '__' + os.path.basename(each_summ_file).split(".")[0]
        df = functools.reduce(lambda left,right: pd.concat([left,right], axis=0),dfs)
        df = df.reset_index()
        df = df.replace(to_replace='RICORD-1c', value="MIDRC_RICORD_1C", regex=True)
        df = df[df.num_images > 0] # drop rows with zero images
        df = df[df['images_info'].map(lambda d: len(d)>0)] # drop rows with no image information
        # # DEBUG - only selecting some patients
        # print("**********DEBUG MODE: not using all patients!**********")
        # df = df.loc[0:99,:]
        original_total_patients = len(df)
        # convert to by-image format
        conversion_tables = {}
        if "gpfs_projects" in args.output_dir:
            conversion_files = conversion_files_openhpc
        elif "scratch" in args.output_dir:
            conversion_files = conversion_files_betsy
        else:
            print("could not find conversion files")
            return
        for c, fp in conversion_files.items():
            if not os.path.exists(fp): # DEBUG
                print(f"No file at {fp}")
                return
            conversion_tables[c] = pd.read_json(fp)
        conv_table = pd.concat(conversion_tables.values())
        df = convert_to_csv_v2(args, df, conv_table)
        if args.limit_classes:
            for key, val in subgroup_dict.items():
                sub = val[1]
                # keep options that are not reported at all
                if len(df[sub].unique()) == 1 and df[sub].unique().tolist()[0] == 'Not Reported':
                    continue
                valid_options = key.replace(" ", "_").split(",")
                valid_options = [item for item in valid_options if item in args.tasks]
                df = df[df[sub].isin(valid_options)]
        for key, val in subgroup_dict.items():
            sub = val[1]
            for k in key.replace(" ","_").split(","):
                # convert from subgroup to binary values
                df[k] = (df[sub] == k).astype(int)

        total_images = len(df)
        total_patients = len(df['patient_id'].unique())
        print(f"\nFound {total_images} images from {total_patients} patients within {len(args.input_list)} repo(s).")
        df.to_csv(overall_csv, index=None)
    
    # 2) generate the individual random seed partition
    # TODO get the repository for each step # [WIP]
    if args.step_repo is None:
        # using all repos provided for all steps
        step_repos = {n:df['repo'].unique() for n in range(args.steps)}
    else:
        step_repos = {}
        for sr in args.step_repo.split("/"):
            # print(sr)
            step_numbers, step_repo = sr.split("__")
            srs = [x for x in step_repo.split(",")]
            # print(srs)
            if "-" in step_numbers: # given a range of numbers
                f, l = step_numbers.split("-")
                step_numbers = [x for x in range(int(f), int(l))]
            else: # given specific step numbers
                step_numbers = [int(x) for x in step_numbers.split(",")]
            # print(step_numbers)
            for x in step_numbers:
                step_repos[x] = srs
    # print(len(df))
    # print(step_repos)
    id_df = df.drop_duplicates(subset=['patient_id'])
    # print(len(id_df))
    if args.stratify != "False":
        strat_classes = []
        for x, y in subgroup_dict.items():
            if y[1] in args.stratify.split(","):
                if len(strat_classes) == 0:
                    strat_classes = x.split(",")
                    continue
                strat_classes = [",".join(z).replace(" ", "_") for z in itertools.product(strat_classes, x.split(","))]
        
        
        print("Strat classes: ", strat_classes)
        strat_id_dfs = {}
        # get the individual df for each class
        min_n = 10000000
        for cls in strat_classes:
            temp_df = id_df.copy()
            for c in cls.split(","):
                temp_df = temp_df[temp_df[c] == 1]
            strat_id_dfs[cls] = temp_df
            if len(temp_df) < min_n:
                min_n = len(temp_df)
        min_n -= subtract_from_smallest_subgroup
        for cls, df in strat_id_dfs.items():
            strat_id_dfs[cls] = df.sample(min_n, random_state=RAND_SEED_INITIAL+args.random_seed)
    
    if args.add_joint_validation != 0:
        df = pd.read_csv(overall_csv, index_col=0)
        if args.stratify != "False":
            jvs = []
            for cls in strat_id_dfs:
                temp_jv = strat_id_dfs[cls].sample(frac=args.add_joint_validation, random_state=RAND_SEED_INITIAL+args.random_seed)
                strat_id_dfs[cls].drop(temp_jv.index, inplace=True)
                jvs.append(temp_jv)
            jv = pd.concat(jvs, axis=0)
        else:
            id_df, jv = train_test_split(id_df, test_size=args.add_joint_validation,random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True)
        v_pids = jv['patient_id'].unique().tolist()
        joint_val = df[df['patient_id'].isin(v_pids)]
        joint_val.to_csv(os.path.join(args.output_dir, 'independent_test.csv'))

    if args.stratify != "False":
        val_samples = {}
        for cls in strat_id_dfs:
            val_samples[cls] = strat_id_dfs[cls].sample(frac=args.percent_test_partition, random_state=RAND_SEED_INITIAL+args.random_seed)
            strat_id_dfs[cls].drop(val_samples[cls].index, inplace=True)
    else:
        tr_sample, val_sample = train_test_split(id_df, test_size=args.percent_test_partition,random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True)

    # generate files
    n_val = int(1/(args.steps))
    step_sizes = get_split_sizes(args)
    print(step_sizes)
    step_tr = {}
    step_val = {}
    for n in range(args.steps):
        print(f"\nstep {n}")
        # validation
        if args.stratify != "False": # currently can only stratify by one binary class
            step_vals = []
            for cls in val_samples:
                temp_val = val_samples[cls].sample(frac=n_val, random_state=RAND_SEED_INITIAL+args.random_seed)
                val_samples[cls].drop(temp_val.index, inplace=True)
                step_vals.append(temp_val)
            step_val[n] = pd.concat(step_vals, axis=0)
        else:
            step_val[n] = val_sample.sample(n=n_val, random_state = RAND_SEED_INITIAL+args.random_seed)
            val_sample.drop(step_val[n].index, axis=0, inplace=True) # remove those patient ids to avoid data leakage
        # get all images for that patient
        df = pd.read_csv(overall_csv)
        v_pids = step_val[n]['patient_id'].unique().tolist()
        validation_split = df[df['patient_id'].isin(v_pids)]
        
        # training
        # TODO: accumulate
        if args.stratify != "False": # currently can only stratify by one binary class
            step_trs = []
            for cls in strat_id_dfs:
                temp_step = strat_id_dfs[cls].sample(frac=step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed)
                strat_id_dfs[cls].drop(temp_step.index, inplace=True)
                step_trs.append(temp_step)
            step_tr[n] = pd.concat(step_trs, axis=0)
        else:
            step_tr[n] = tr_sample.sample(n=int(step_sizes[n]/2), random_state = RAND_SEED_INITIAL+args.random_seed)
            tr_sample.drop(step_tr[n].index, axis=0, inplace=True) # remove those patient ids to avoid data leakage
        if args.accumulate > 0 and n >0: # TODO: stratify and accumulate
            if args.stratify != "False": # currently can only stratify by one binary class
                acc = step_tr[n-1].groupby(args.stratify, group_keys=False).apply(lambda x: x.sample(frac=args.accumulate, random_state = RAND_SEED_INITIAL+args.random_seed))
            else:
                acc = step_tr[n-1].sample(frac=args.accumulate, random_state = RAND_SEED_INITIAL+args.random_seed)
            step_tr[n] = pd.concat([step_tr[n], acc],axis=0)
        # get all images for that patient
        df = pd.read_csv(overall_csv)
        tr_pids = step_tr[n]['patient_id'].unique().tolist()
        training_split = df[df['patient_id'].isin(tr_pids)]
        
        # save files
        training_split.to_csv(os.path.join(args.output_dir, f"step_{n}.csv"))
        validation_split.to_csv(os.path.join(args.output_dir, f"step_{n}_validation.csv"))

   
        
    

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-i', '--input_list', action='append', help='<Required> List of input summary files', required=True, default=[])
    parser.add_argument('-o', '--output_dir', help='<Required> output dir to save list files', required=True)
    parser.add_argument('-r', '--random_seed', help='random seed for experiment reproducibility (default = 2020)', default=2020, type=int)
    parser.add_argument('-p', '--percent_test_partition', help='percent test partition (default = 0.2)', default=0.2, type=float)
    parser.add_argument('-s', '--select_option', help='type of partition (default = 0)', default=0, type=int)
    parser.add_argument('-m', '--min_num_image_per_patient', help='min number of images per patient (default = 1)', default=1, type=int)

    # parser.add_argument('-strat_groups', default='F-DX,F-CR,M-CR,M-DX')
    parser.add_argument('-stratify', default=False)
    parser.add_argument('-steps', default=2, type=int)
    parser.add_argument('-split_type', default="equal")
    parser.add_argument('-tasks', default="F,M,CR,DX")
    # parser.add_argument('-tasks', default="auto")
    # parser.add_argument("-accumulate", default=False)
    parser.add_argument('-accumulate', default=0, type=float)
    parser.add_argument("-add_joint_validation", default=0, type=float)
    # parser.add_argument("-limit_classes", default=False, help='Only include samples that belong to one to one of the tasks specified')
    # # DEBUG
    parser.add_argument("-limit_classes", default=True, help='Only include samples that belong to one to one of the tasks specified')
    parser.add_argument("-partition_name", required=True, help="name associated with the partition generation settings")
    parser.add_argument("-step_repo", help="repository for each step #")

    args = parser.parse_args()
    # # call
    output_log_file = os.path.join(args.output_dir, 'tracking.log')
    bootstrapping_v2(args)
    # bootstrapping(args)
    tracking_info = {"Partition":args.__dict__}
    tracking_info['Partition']["Generated on"] = str(date.today())
    with open(output_log_file, 'w') as fp:
        json.dump(tracking_info, fp, indent=2)
    print("Done")
