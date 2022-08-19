'''
    Program that partitions data based on the input summary json files
'''
import os
import argparse
import json
import functools
import pandas as pd
from sklearn.model_selection import train_test_split

subtract_from_smallest_subgroup = 5
RAND_SEED_INITIAL=2022


def stratified_bootstrapping_default(args):
    '''
        default subgroups: FDX, FCR, MDC, MCR
        call this funtion repeatedly with different random seed
        to perform stratified boostrapping

        By default tries to maintain the ratio of these subgroups within
        the tr or ts: sex, modality, repo

        Note that this method does not match the number images/patient across subgroups
    '''
    dfs = []
    repo_str = ''
    for each_summ_file in args.input_list:
        dfs += [pd.read_json(each_summ_file, orient='table')]
        repo_str += '__' + os.path.basename(each_summ_file).split('.')[0]
    # out_file_str = os.path.join(args.output_dir, repo_str)
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
    # print([len(df_FDX.index), len(df_FCR.index), len(df_MDX.index), len(df_MCR.index)])
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
    stratified_sample[0].to_json(out_fname, indent=4, orient='table', index=False)
    print(stratified_sample[0].shape)

    # # ts partition
    tr_patient_ids = list(stratified_sample[0]['patient_id'])
    # print(tr_patient_ids)
    new_ts_df = df[~df.patient_id.isin(tr_patient_ids)]
    print('\n>>> PARTITION #{} with {} patients'.format(1, new_ts_df.shape[0]))
    # print(new_ts_df.shape)
    print(new_ts_df.groupby("sex")['modality'].value_counts())
    print(new_ts_df.groupby("sex")['COVID_positive'].value_counts())
    print(new_ts_df.groupby("sex")['repo'].value_counts())
    out_fname = os.path.join(args.output_dir, 'ts' + repo_str + '.json')
    new_ts_df.to_json(out_fname, indent=4, orient='table', index=False)
    # ts_remaining

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-i', '--input_list', action='append', help='<Required> List of input summary files', required=True, default=[])
    parser.add_argument('-o', '--output_dir', help='<Required> output dir to save list files', required=True)
    parser.add_argument('-r', '--random_seed', help='random seed for experiment reproducibility (default = 2020)', default=2020, type=int)
    parser.add_argument('-p', '--percent_test_partition', help='percent test partition (default = 0.2)', default=0.2, type=float)
    args = parser.parse_args()
    # # call
    output_log_file = os.path.join(args.output_dir, 'log.log')
    with open(output_log_file, 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
        stratified_bootstrapping_default(args)
