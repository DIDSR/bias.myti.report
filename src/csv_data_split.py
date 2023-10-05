import pandas as pd
import os
import argparse
from sklearn.utils import shuffle

subgroup_list = ["F", "Black"]

def data_sampling(in_data, subs=1.0, prev=None, test_subgroup='F'):
    """ Sample the data according to subsample rate and prevalence ratio.
        Subsample rate be the same in each subgroup combination.
    """
    out = pd.DataFrame()
    remain_subgroup = [i for i in subgroup_list if i != test_subgroup]
    #equally sampling in each subgroup combinations
    for i in range(2):
      for j in range(2):
        for k in range (2):
          data_temp = in_data[(in_data[test_subgroup] == i)&(in_data[remain_subgroup[0]] == j)&(in_data['Yes'] == k)]
          data_temp = data_temp.sample(frac=subs)
          if prev is not None:
              if i == k:
                data_temp = data_temp.sample(frac=prev)
              else:
                data_temp = data_temp.sample(frac=1-prev)
          out = out.append(data_temp)           
    return shuffle(out)     


def train_split(args):
    """ Subsample the dataset by the input sampling rate. Manipulate disease prevalence in different subgroups.
    """
    #read input data csv files
    input_data = pd.read_csv(os.path.join(args.in_dir, args.input_file))        
    subs = args.subsample 
    prev = args.prevalence
    test_subgroup = args.test_subgroup 
    if prev is None:
    #only subsampling no prevalence
        print(f"Start data subsampling with rate of {subs}\n")
        out_data = data_sampling(input_data, subs=subs)
        out_data.to_csv(os.path.join(args.save_dir, f"{round(subs*100)}subsample_{args.input_file}"), index=False)
    else:
        if test_subgroup not in subgroup_list:
            raise RuntimeError('Input test subgroups is not currently supported!')        
        if subs == 1.0:
        #only manipulate prevalence in each subgroup
            print(f"Start data split of {prev} for {test_subgroup}\n")
            out_data = data_sampling(input_data, prev=prev, test_subgroup=test_subgroup)
            out_data.to_csv(os.path.join(args.save_dir, f"{round(prev*100)}{test_subgroup[0]}P_{args.input_file}"), index=False)
        else:
        #do both subsampling and prevalence manipulation
            print(f"Start data split of {prev} for {test_subgroup} with subsampling rate of {subs}\n")
            out_data = data_sampling(input_data, subs=subs, prev=prev, test_subgroup=test_subgroup)
            out_data.to_csv(os.path.join(args.save_dir, f"{round(subs*100)}subsample_{round(prev*100)}{test_subgroup[0]}P_{args.input_file}"), index=False)
   
    
    
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prevalence',default=None,type=float)
    parser.add_argument('--subsample',default=1.0,type=float)
    parser.add_argument('--test_subgroup',type=str,help="Current choices: F or Black")
    parser.add_argument('--in_dir',type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--input_file',type=str)
    args = parser.parse_args()
    train_split(args)