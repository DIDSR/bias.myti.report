import pandas as pd
import os
import argparse
from sklearn.utils import shuffle

subgroup_list = ["F", "Black"]
     
def train_split(args):
    """ 
    Manipulate prevalence in race or sex subgroups. The given subgroup will be sampled to the specified prevalence, 
    while the opposite subgroup will be sampled to (1-prevalence).
    
    """
    #read input data csv files
    input_data = pd.read_csv(os.path.join(args.in_dir, args.input_file)) 
    test_subgroup = args.test_subgroup       
    for prev in args.prevalences:         
        if test_subgroup not in subgroup_list:
            raise RuntimeError('Input test subgroups is not currently supported!')        
        #only manipulate prevalence in each subgroup
        else:    
            print(f"Start data split of prevalence of {prev} for {test_subgroup}\n")
            out = pd.DataFrame()
            remain_subgroup = [i for i in subgroup_list if i != test_subgroup]
            #equally sampling in each subgroup combinations
            for i in range(2):
              for j in range(2):
                for k in range (2):
                  data_temp = input_data[(input_data[test_subgroup] == i)&(input_data[remain_subgroup[0]] == j)&(input_data['Yes'] == k)]
                  if prev is not None:
                      if i == k:
                        data_temp = data_temp.sample(frac=prev)
                      else:
                        data_temp = data_temp.sample(frac=1-prev)
                  out = pd.concat([out, data_temp])          
            out_data = shuffle(out)     
            out_data.to_csv(os.path.join(args.save_dir, f"{round(prev*100)}{test_subgroup[0]}P_{args.input_file}"), index=False)
   
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prevalences', nargs='+', default=[], required=True, type=float,
    help="List of prevalences to be sampled in the given subgroup, while prevalence in the opposite subgroup will be (1-prevalence).")
    parser.add_argument('--test_subgroup', type=str, help="Current choices: F or Black")
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    train_split(args)