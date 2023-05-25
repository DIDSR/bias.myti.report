import pandas as pd
import os
import argparse
from sklearn.utils import shuffle

def train_split():
    """
    split the training/validation dataset evenly for each subgroup (sex, race and COVID)
    """
    #read input csv files
    train = pd.read_csv(os.path.join(args.main_dir, args.input_file))
    train_1_out = pd.DataFrame()
    train_2_out = pd.DataFrame()
    #concat evenly splitted data in each subgroup
    for m in range(2):
      for w in range(2):
        for y in range (2):
          tr_temp = train[(train['M'] == m)&(train['White'] == w)&(train['Yes'] == y)]
          tr_temp_1 = tr_temp.sample(frac=0.5)
          tr_temp_2 = tr_temp.drop(tr_temp_1.index)
          train_1_out = train_1_out.append(tr_temp_1)
          train_2_out = train_2_out.append(tr_temp_2)       
    #output 2 csv files     
    train_1_out = shuffle(train_1_out)
    train_1_out.to_csv(os.path.join(args.dest_dir, args.output_1), index=False)
    train_2_out = shuffle(train_2_out)
    train_2_out.to_csv(os.path.join(args.dest_dir, args.output_2), index=False)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rand',default=5,type=int)
    parser.add_argument('--batch',default=5,type=int)
    parser.add_argument('--in_dir',type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--output_1',type=str)
    parser.add_argument('--output_2',type=str)
    args = parser.parse_args()
    train_split()