import os
import pandas as pd
import argparse
import json
from constants import *
import numpy as np


def summarize():
    layer = args.layer
    file_name = args.file_name
    main_dir = args.main_dir
    summary_file = []
    for rand in range(args.rand_num):
        fairness = pd.read_csv(os.path.join(main_dir, f'RAND_{rand}', args.folder_name, args.file_name_1))
        nuance = pd.read_csv(os.path.join(main_dir, f'RAND_{rand}', args.folder_name, args.file_name_2))
        combined  = fairness.join(nuance, lsuffix='_1', rsuffix='_2')
        summary_file.append(combined)
    summary = pd.concat(summary_file)
    summary.to_csv(os.path.join(main_dir, f'{layer}_overall_summary_{file_name}.csv'), index=False)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer',type=str)
    parser.add_argument('--main_dir',type=str)
    parser.add_argument('--folder_name',type=str)
    parser.add_argument('--file_name',type=str)
    parser.add_argument('--file_name_1',type=str)
    parser.add_argument('--file_name_2',type=str)
    parser.add_argument('--rand_num',type=int,default=5)
    args = parser.parse_args()

    summarize()
    print("Done\n")
