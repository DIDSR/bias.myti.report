'''
    Program that partitions data based on the input summary json files
'''
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

subtract_from_smallest_subgroup = 5
RAND_SEED_INITIAL=2022

subgroup_dict = {
    "F,M":["patient_info","sex"],
    "CR,DX":["images_info","modality"],
    "Yes,No":["patient_info","COVID_positive"],
    "Asian,Black or African American,White":["patient_info","race"]
}
conversion_files = {repo:f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/{repo}_jpegs/conversion_table.json" for repo in ["MIDRC_RICORD_1C", "COVID_19_NY_SBU", "COVID_19_AR", "open_AI", "open_RI"]}

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
        args.tasks = args.tasks.replace("_", " ")
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
    for c, fp in conversion_files.items():
        conversion_tables[c] = pd.read_json(fp)
    # deal with any settings that can reduce the number of possible samples available
    new_df = get_n_patients(args, df)
    if args.stratify != "False":
        # new_df, min_subgroup_size, strat_idxs = get_n_patients(args,df)
        tr_sample, _ = train_test_split(new_df, test_size=args.percent_test_partition, random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True, stratify=new_df[args.stratify])
    else:
        # new_df = get_n_patients(args, df)
        tr_sample, _ = train_test_split(new_df, test_size=args.percent_test_partition, random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True)
    # separate + save validation split (uses all patients not used in the training sample, not caring for stratification/min image)
    tr_idxs = tr_sample.index.tolist()
    val_idxs =[idx for idx in df.index.tolist() if idx not in tr_idxs]
    val_split = df.loc[val_idxs]
    # validation bootstrapping:
    # TODO: allow unequal steps (?)
    val_n = int(len(val_split)/args.steps)
    for n in range(args.steps):
        step_val = val_split.sample(n=val_n, random_state=RAND_SEED_INITIAL+args.random_seed)
        step_val.to_json(os.path.join(args.output_dir, f"step_{n}_validation.json"), orient='table', indent=2)
        step_csv = convert_to_csv(args, step_val, conversion_tables)
        step_csv.to_csv(os.path.join(args.output_dir, f"step_{n}_validation.csv"))

    # val_split.to_json(os.path.join(args.output_dir, "validation.json"), orient='table', indent=2)
    # val_csv = convert_to_csv(args, val_split, conversion_tables)
    # val_csv.to_csv(os.path.join(args.output_dir, "validation.csv"))
    
    step_sizes = get_split_sizes(args, len(tr_sample))
    # print(step_sizes)
    step_dfs = {}
    # separate individual steps
    for n in range(args.steps):
        if n == args.steps-1:
            step_dfs[n] = tr_sample.sample(n=step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed)
        else:
            if args.stratify != "False":
                _, step_dfs[n] = train_test_split(tr_sample, test_size = step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True, stratify=tr_sample[args.stratify])
            else:
                _, step_dfs[n] = train_test_split(tr_sample, test_size = step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True)
            # step_dfs[n] = tr_sample.sample(n=step_sizes[n], random_state=RAND_SEED_INITIAL+args.random_seed)
            tr_sample.drop(step_dfs[n].index, axis=0, inplace=True)
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
    csv_df = pd.DataFrame(columns=["Path"]+[task for task in args.tasks])
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
            for task in args.tasks:
                for key, val in subgroup_dict.items():
                    if task in key.split(','):
                        tval = val
                if 'patient_info' in tval:
                    img_info[task] = int(row[tval[0]][0][tval[1]] == task)
                else:
                    img_info[task] = int(row[tval[0]][ii][tval[1]] == task)
            csv_df.loc[len(csv_df)] = img_info
    return csv_df

# def sample_steps(args, split_sizes, input_df):
#     dfs = {}
#     for i in range(args.steps):
#         if i == args.steps-1:
#             dfs[f"step {i}"] = input_df
#         else:
#             dfs[f"step {i}"] = input_df.sample(split_sizes[f"step {i}"], random_state=RAND_SEED_INITIAL+args.random_seed)
#             input_df.drop(dfs[f"step {i}"].index, axis=0, inplace=True)
#         if args.accumulate == True and i > 0 and i != args.steps-1:
#             dfs[f"step {i}"] = pd.concat([dfs[f"step {i}"], dfs[f"step {i-1}"]])
#     return dfs

def get_split_sizes(args, total_number):
    split_sizes = [0 for i in range(args.steps)]
    if args.split_type == 'equal':
        for i in range(args.steps):
            split_sizes[i] = int((1/args.steps) * total_number)
    elif args.split_type == 'increasing':
        tot_splits = (args.steps*args.steps + args.steps)/2
        for i in range(args.steps):
            split_sizes[i] = int((1/tot_splits)*(i+1)*total_number)   
    elif args.split_type == 'random':
        random.seed(RAND_SEED_INITIAL+args.random_seed)
        x = random.sample(range(1,10), args.steps)
        y = [z/sum(x) for z in x]
        for i in range(args.steps):
            split_sizes[i] = int(y[i]*total_number)
    elif type(args.split_type) == list:
        for i in range(args.steps):
            split_sizes[i] = int(args.split_type[i]*total_number)
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
    parser.add_argument("-accumulate", default=False)
    args = parser.parse_args()
    # # call
    output_log_file = os.path.join(args.output_dir, 'tracking.log')
    bootstrapping(args)
    tracking_info = {"Partition":args.__dict__}
    tracking_info['Partition']["Generated on"] = str(date.today())
    tracking_info['Models'] = {}
    with open(output_log_file, 'w') as fp:
        json.dump(tracking_info, fp, indent=2)
