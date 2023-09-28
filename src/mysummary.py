import os
import pandas as pd
import argparse
import numpy as np


def summarize(args):
    '''
    summarize all the results from experiments in different batches and rand_seeds
    calculate the mean and standard deviation per intended test subgroup
    results save as a csv file
    '''
    exp_name = args.exp_name
    main_dir = args.main_dir
    # # gathering all the results from different batches and random_seeds
    all_file = []
    for batch in range(args.batch_num):
        for rand in range(args.rand_num):
            temp_file = pd.read_csv(os.path.join(main_dir, f'batch_{batch}/RAND_{rand}/{exp_name}_RD_0', args.input_file))
            all_file.append(temp_file)
    all_result = pd.concat(all_file)
    # # calculate mean and std per each subgroup
    summary = pd.DataFrame(columns=all_result.columns.values.tolist())
    for grp in args.test_subgroup:
        # check if the result df contains the subgroup
        if grp not in all_result.iloc[:,0].values:
            print(f'Input test subgroup {grp} is not in result file, skipping')
        else:           
            grp_dp = all_result.loc[all_result.iloc[:,0] == grp]
            summary.loc[f'{grp}_mean'] = grp_dp.mean()
            summary.loc[f'{grp}_std'] = grp_dp.std()
    summary.to_csv(os.path.join(main_dir, args.output_file), index=True)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',type=str)
    parser.add_argument('--main_dir',type=str)
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--output_file',type=str)
    parser.add_argument('--test_subgroup',nargs='+',type=str)
    parser.add_argument('--rand_num',type=int,default=1)
    parser.add_argument('--batch_num',type=int,default=5)
    args = parser.parse_args()
    print("Start summarization\n")
    summarize(args)
    print("Done\n")
