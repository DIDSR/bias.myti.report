import pandas as pd
import os
import argparse
from sklearn.utils import shuffle

custom_subgroups ={ 
    'sex':{'F','M'},
    'race':{'Black', 'White'},
}

def train_split():
    """
    split the training/validation dataset evenly for each subgroup (sex, race and COVID)
    """
    #read input csv files
    train = pd.read_csv(os.path.join(args.in_dir, args.input_file))
    train_1_out = pd.DataFrame()
    train_2_out = pd.DataFrame()
    #concat data splitted by the input fraction in each subgroup
    keysList = list(custom_subgroups.keys())
    grps = list(custom_subgroups[args.test_subgroup])
    remain_key = [keys for keys in keysList if keys != args.test_subgroup]
    grps_remain = list(custom_subgroups[remain_key[0]])
    for i in range(2):
      for j in range(2):
        for k in range (2):
          tr_temp = train[(train[grps[0]] == i)&(train[grps_remain[0]] == j)&(train['Yes'] == k)]
          if i == k:
            tr_temp_1 = tr_temp.sample(frac=args.fraction)
          else:
            tr_temp_1 = tr_temp.sample(frac=1-args.fraction)
          tr_temp_2 = tr_temp.drop(tr_temp_1.index)
          train_1_out = train_1_out.append(tr_temp_1)
          train_2_out = train_2_out.append(tr_temp_2)       
    #output 2 csv files     
    train_1_out = shuffle(train_1_out)
    train_1_out.to_csv(os.path.join(args.save_dir, args.output_1), index=False)
    train_2_out = shuffle(train_2_out)
    train_2_out.to_csv(os.path.join(args.save_dir, args.output_2), index=False)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraction',default=0.5,type=float)
    parser.add_argument('--test_subgroup',type=str)
    parser.add_argument('--in_dir',type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--output_1',type=str)
    parser.add_argument('--output_2',type=str)
    args = parser.parse_args()
    train_split()