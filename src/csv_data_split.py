import pandas as pd
import os
import argparse
from sklearn.utils import shuffle


def train_split(args):
    """ Split the training/validation dataset according to fraction.
    """
    #read input csv files
    train = pd.read_csv(os.path.join(args.in_dir, args.input_file))
    train_1_out = pd.DataFrame()
    train_2_out = pd.DataFrame()
    test_subgroup = args.test_subgroup
    frac = args.fraction
    print(f"Start data split of {frac} and {1-frac} for {test_subgroup[0]} and {test_subgroup[1]}\n")
    #concat data splitted by the input fraction in each subgroup
    subgroup_list = ["F", "M", "Black", "White"]
    remain_subgroup = list(set(subgroup_list) - set(test_subgroup))
    for i in range(2):
      for j in range(2):
        for k in range (2):
          tr_temp = train[(train[test_subgroup[0]] == i)&(train[remain_subgroup[0]] == j)&(train['Yes'] == k)]
          if i == k:
            tr_temp_1 = tr_temp.sample(frac=frac)
          else:
            tr_temp_1 = tr_temp.sample(frac=1-frac)
          tr_temp_2 = tr_temp.drop(tr_temp_1.index)
          train_1_out = train_1_out.append(tr_temp_1)
          train_2_out = train_2_out.append(tr_temp_2)       
    #output 2 csv files     
    train_1_out = shuffle(train_1_out)
    train_1_out.to_csv(os.path.join(args.save_dir, f"{round(frac*100)}{test_subgroup[0][0]}P_{args.input_file}"), index=False)
    train_2_out = shuffle(train_2_out)
    train_2_out.to_csv(os.path.join(args.save_dir, f"{round((1-frac)*100)}{test_subgroup[0][0]}P_{args.input_file}"), index=False)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction',default=0.5,type=float)
    parser.add_argument('--test_subgroup',nargs=2,type=str,help="Current choices: F M or Black White")
    parser.add_argument('--in_dir',type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--input_file',type=str)
    args = parser.parse_args()
    train_split(args)