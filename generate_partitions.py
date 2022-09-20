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
import math

subtract_from_smallest_subgroup = 5
RAND_SEED_INITIAL=2022

subgroup_dict = {
    "F,M":["patient_info","sex"],
    "CR,DX":["images_info","modality"]
}


# TODO: automatic selection by repository
conversion_file = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/MIDRC_RICORD_1C_jpegs/conversion_table.json"

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
    if args.stratify != "True": # no stratifying 
        split_sizes = get_split_sizes(args, len(df))
        dfs = sample_steps(args, split_sizes=split_sizes, input_df=df)
    else: # stratifyng
        print("stratifying")
        # TODO: prevent stratifying on last step (validation), instead take all remaining samples of each 
        # get sub_dfs
        sub_dfs = {}
        for s in subgroup_dict.values():
            sub_dfs[s[1]] = df.apply(lambda row:row[s[0]][0][s[1]], axis=1)
        args.strat_groups = args.strat_groups.split(",")
        strat_dfs = {}
        for strat_group in args.strat_groups:
            sub_groups = strat_group.split("-")
            sub_idxs = {sub:None for sub in sub_groups}
            for sub in sub_groups:
                for key, val in subgroup_dict.items():
                    if sub not in key.split(","):
                        continue
                    sub_idxs[sub] = sub_dfs[val[1]][sub_dfs[val[1]]==sub].index.tolist()
            idx = find_overlap(sub_idxs)
            strat_dfs[strat_group] = df.loc[idx]

        min_subgroup_size = min([len(x) for x in strat_dfs.values()]) - subtract_from_smallest_subgroup
        # now divide the steps
        split_sizes = get_split_sizes(args, min_subgroup_size)
        ss_dfs = {}
        for s, sdf in strat_dfs.items():
            ss_dfs[s] = sample_steps(args, split_sizes, sdf)
        dfs = {}
        for i in range(args.steps):
            dfs[f'step {i}'] = pd.concat([ss_dfs[strat][f"step {i}"] for strat in strat_dfs])
    # image number options
    if args.select_option == 1:
        for i in range(args.steps-1): # don't apply image restriction to the test step
            dfs[f"step {i}"] = select_image_per_patient(dfs[f"step {i}"], args.min_num_image_per_patient)
    # saving + prining info
    print(f"\nAccumulate: ", args.accumulate)
    print(f"Stratify: ", args.stratify)
    conversion_table = pd.read_json(conversion_file)
    tasks = args.tasks.split(",")
    for i in range(args.steps):
        print(f"Step {i} with {len(dfs[f'step {i}'])} patients")
        #TODO: print out stratified info (method from stratieifed_bootstrapping doesn't work)
        out_fname = os.path.join(args.output_dir, f"step_{i}_{repo_str}.json")
        dfs[f"step {i}"].to_json(out_fname, indent=4, orient='table', index=False)
        # convert to csv (as the model requires csv input, by image rather than by patient)
        csv_df = pd.DataFrame(columns=["Path"]+[task for task in tasks])
        for iii, row in dfs[f"step {i}"].iterrows():
            for ii, img in enumerate(row['images']):
                img_info = {}
                # get the jpeg path
                img_info["Path"] = conversion_table[conversion_table['dicom']==img]['jpeg'].values[0]
                
                for task in tasks:
                    for key, val in subgroup_dict.items():
                        if task in key.split(','):
                           tval = val
                    if 'patient_info' in tval:
                        img_info[task] = int(row[tval[0]][0][tval[1]] == task)
                    else:
                        img_info[task] = int(row[tval[0]][ii][tval[1]] == task)
            csv_df.loc[len(csv_df)] = img_info
        csv_df.to_csv(os.path.join(args.output_dir, f"step_{i}.csv"))

def sample_steps(args, split_sizes, input_df):
    dfs = {}
    for i in range(args.steps):
        dfs[f"step {i}"] = input_df.sample(split_sizes[f"step {i}"], random_state=RAND_SEED_INITIAL+args.random_seed)
        input_df.drop(dfs[f"step {i}"].index, axis=0, inplace=True)
        if args.accumulate == True and i > 0 and i != args.steps-1:
            dfs[f"step {i}"] = pd.concat([dfs[f"step {i}"], dfs[f"step {i-1}"]])
        if args.stratify != "True":
            print(f"Step {i}: {len(dfs[f'step {i}'])} samples")
    return dfs

def get_split_sizes(args, total_number):
    split_sizes = {}
    print(f"Loaded {total_number} samples \nsplitting into {args.steps} steps with {args.split_type} sizes")
    if args.split_type == 'equal':
        for i in range(args.steps):
            split_sizes[f"step {i}"] = int((1/args.steps) * total_number)
    elif args.split_type == 'increasing':
        tot_splits = (args.steps*args.steps + args.steps)/2
        for i in range(args.steps):
            split_sizes[f"step {i}"] = int((1/tot_splits)*(i+1)*total_number)   
    elif args.split_type == 'random':
        random.seed(RAND_SEED_INITIAL+args.random_seed)
        x = random.sample(range(1,10), args.steps)
        y = [z/sum(x) for z in x]
        for i in range(args.steps):
            split_sizes[f"step {i}"] = int(y[i]*total_number)
    elif type(args.split_type) == list:
        for i in range(args.steps):
            split_sizes[f"step {i}"] = int(args.split_type[i]*total_number)
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
            
def stratified_bootstrapping_default(args):
    '''
        default subgroups: FDX, FCR, MDC, MCR
        call this funtion repeatedly with different random seed
        to perform stratified boostrapping

        By default tries to maintain the ratio of these subgroups within
        the tr or ts: sex, modality, repo

        Note that this method does not match the number images/patient across subgroups
        If the option=1 is selected, then only training set is process to the choice of
        number of images/patient based on the sorted order of the "study date", i.e.,
        earlier study dates are seleted first.
    '''
    dfs = []
    repo_str = ''
    for each_summ_file in args.input_list:
        dfs += [pd.read_json(each_summ_file, orient='table')]
        repo_str += '__' + os.path.basename(each_summ_file).split('.')[0]
    df = functools.reduce(lambda left,right: pd.concat([left, right], axis=0),dfs)
    df = df.reset_index()
    print(df.shape)
    # df.drop(['bad_images_info', 'bad_images'], axis=1, inplace=True)
    df = df[df.num_images > 0]  # # drop rows with 0 images
    df['sex'] = df.apply(lambda row: row.patient_info[0]['sex'], axis = 1)
    df['modality'] = df.apply(lambda row: row.images_info[0]['modality'], axis = 1)
    df['COVID_positive'] = df.apply(lambda row: row.patient_info[0]['COVID_positive'], axis = 1)
    # #
    print('\n>>> Original distribution')
    print(df.groupby("sex")['modality'].value_counts())
    # #
    df_FDX = df.loc[(df['sex'] == 'F') & (df['modality'] == 'DX')]
    df_FCR = df.loc[(df['sex'] == 'F') & (df['modality'] == 'CR')]
    df_MDX = df.loc[(df['sex'] == 'M') & (df['modality'] == 'DX')]
    df_MCR = df.loc[(df['sex'] == 'M') & (df['modality'] == 'CR')]
    min_subgroup_size = min(len(df_FDX.index), len(df_FCR.index), len(df_MDX.index), len(df_MCR.index)) - subtract_from_smallest_subgroup
    # # set RANDOM SEED here
    new_df = pd.concat([df_FDX.sample(n=min_subgroup_size, random_state=RAND_SEED_INITIAL+args.random_seed), 
                        df_FCR.sample(n=min_subgroup_size, random_state=RAND_SEED_INITIAL+args.random_seed), 
                        df_MDX.sample(n=min_subgroup_size, random_state=RAND_SEED_INITIAL+args.random_seed), 
                        df_MCR.sample(n=min_subgroup_size, random_state=RAND_SEED_INITIAL+args.random_seed)], axis=0)
    # # set RANDOM SEED here
    stratified_sample = train_test_split(new_df, test_size=args.percent_test_partition, random_state=RAND_SEED_INITIAL+args.random_seed, shuffle=True, stratify=new_df[['sex', 'modality']])
    # # tr partition
    print('\n>>> PARTITION #{} with {} patients'.format(0, stratified_sample[0].shape[0]))
    print(stratified_sample[0].groupby("sex")['modality'].value_counts())
    print(stratified_sample[0].groupby("sex")['COVID_positive'].value_counts())
    print(stratified_sample[0].groupby("sex")['repo'].value_counts())
    out_fname = os.path.join(args.output_dir, 'tr' + repo_str + '.json')
    if args.select_option == 0:
        stratified_sample[0].to_json(out_fname, indent=4, orient='table', index=False)
        print(stratified_sample[0].shape)
    else:
        # # select imaeg/patient before saving to json
        cleaned_up_df = select_image_per_patient(stratified_sample[0], args.min_num_image_per_patient)
        cleaned_up_df.to_json(out_fname, indent=4, orient='table', index=False)
        print(cleaned_up_df)
    

    # # ts partition
    tr_patient_ids = list(stratified_sample[0]['patient_id'])
    new_ts_df = df[~df.patient_id.isin(tr_patient_ids)]
    print('\n>>> PARTITION #{} with {} patients'.format(1, new_ts_df.shape[0]))
    print(new_ts_df.groupby("sex")['modality'].value_counts())
    print(new_ts_df.groupby("sex")['COVID_positive'].value_counts())
    print(new_ts_df.groupby("sex")['repo'].value_counts())
    out_fname = os.path.join(args.output_dir, 'ts' + repo_str + '.json')
    new_ts_df.to_json(out_fname, indent=4, orient='table', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-i', '--input_list', action='append', help='<Required> List of input summary files', required=True, default=[])
    parser.add_argument('-o', '--output_dir', help='<Required> output dir to save list files', required=True)
    parser.add_argument('-r', '--random_seed', help='random seed for experiment reproducibility (default = 2020)', default=2020, type=int)
    # parser.add_argument('-p', '--percent_test_partition', help='percent test partition (default = 0.2)', default=0.2, type=float)
    parser.add_argument('-s', '--select_option', help='type of partition (default = 0)', default=0, type=int)
    parser.add_argument('-m', '--min_num_image_per_patient', help='min number of images per patient (default = 1)', default=1, type=int)

    parser.add_argument('-strat_groups', default='F-DX,F-CR,M-CR,M-DX')
    parser.add_argument('-stratify', default=False)
    parser.add_argument('-steps', default=2, type=int)
    parser.add_argument('-split_type', default="equal")
    parser.add_argument('-tasks', default="F,M,CR,DX")
    parser.add_argument("-accumulate", default=False)
    args = parser.parse_args()
    # # call
    output_log_file = os.path.join(args.output_dir, 'log.log')
    bootstrapping(args)
    with open(output_log_file, 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    #     stratified_bootstrapping_default(args)
