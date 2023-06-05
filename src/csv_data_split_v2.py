import pandas as pd
import os
import argparse
from sklearn.utils import shuffle

custom_subgroups ={ 
    'sex':{'F','M'},
    'race':{'Black', 'White'},
    'modality':{'CR', 'DX'}
}

def train_split_2():
    """
    split the training/validation dataset evenly for each subgroup (sex, race and COVID)
    """
    #read input csv files
    train = pd.read_csv(os.path.join(args.in_dir, args.input_file))
    train_1_out = pd.DataFrame()
    train_2_out = pd.DataFrame()
    grps = list(custom_subgroups[args.test_subgroup])
    #split data according to interested subgroups
    tr_temp_1 = train[(train[grps[0]] == 1)&(train['Yes'] == 1)]
    train_1_out = train_1_out.append(tr_temp_1)
    tr_temp_2 = train[(train[grps[1]] == 1)&(train['Yes'] == 0)]
    train_1_out = train_1_out.append(tr_temp_2)
    tr_temp_3 = train[(train[grps[1]] == 1)&(train['Yes'] == 1)]
    train_2_out = train_2_out.append(tr_temp_3)
    tr_temp_4 = train[(train[grps[0]] == 1)&(train['Yes'] == 0)]
    train_2_out = train_2_out.append(tr_temp_4)
     
    #output 2 csv files     
    train_1_out = shuffle(train_1_out)
    train_1_out.to_csv(os.path.join(args.save_dir, args.output_1), index=False)
    train_2_out = shuffle(train_2_out)
    train_2_out.to_csv(os.path.join(args.save_dir, args.output_2), index=False)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand',default=5,type=int)
    parser.add_argument('--batch',default=5,type=int)
    parser.add_argument('--test_subgroup',type=str)
    parser.add_argument('--in_dir',type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--output_1',type=str)
    parser.add_argument('--output_2',type=str)
    args = parser.parse_args()
    train_split_2()