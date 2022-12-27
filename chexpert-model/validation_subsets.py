import pandas as pd
import json

# VARIABLES
# validation_file = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/1_step_all_CR_stratified_ind_test/RAND_0/step_0_validation.csv"
split_sizes = [1.0, 0.90, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# strat_groups = [("F","Black_or_African_American","Yes"), ("F","Black_or_African_American","No"),
#                 ("M","Black_or_African_American","Yes"), ("M","Black_or_African_American","No"),
#                 ("F","White","Yes"), ("F","White","No"),
#                 ("M","White","Yes"), ("M","White","No")]
strat_groups = [("F","Black","Yes"), ("F","Black","No"),
                ("M","Black","Yes"), ("M","Black","No"),
                ("F","White","Yes"), ("F","White","No"),
                ("M","White","Yes"), ("M","White","No")]
# reduction_groups = [("F"), ("M")]
reduction_groups = None
# CONSTANTS
abbreviation_table = {
    'Female':"F",
    'Male':"M",
    'CR':"C",
    "DX":"D",
    "White":"W",
    'Black':"B",
    "Yes":"P",# positive
    "No":'N'
}

def divide_validation(input_file, sizes=split_sizes, stratification_subgroups=strat_groups, reduce_by=reduction_groups, random_state=0):
    # set up
    dict_fp  = input_file.replace("validation.csv", "validation_files.json")
    overall_dict = {"random_state":random_state} # TODO: load from file ?
    df = pd.read_csv(input_file)
    bp_df = df.drop(['Path'], axis=1).drop_duplicates() # by-patient
    sub_dfs = {}
    for subg in stratification_subgroups:
        temp_df = bp_df.copy()
        for s in subg:
            temp_df = temp_df[temp_df[s] == 1]
        sub_dfs[subg] = temp_df.copy()

    for size in sizes:
        # set up dictionary
        split_id = f"{int(size*100)}_equal"
        split_fp = input_file.replace("validation.csv", f"{split_id}_validation.csv")
        overall_dict[split_id] = {'overall_split_portion':size, 'subgroup_portions':{str(i):size for i in stratification_subgroups}, 'file':split_fp}
        # TODO: non-equal
        split_dfs = []
        for subg in stratification_subgroups:
            temp_df = sub_dfs[subg].sample(frac = overall_dict[split_id]['subgroup_portions'][str(subg)], random_state=random_state)
            split_dfs.append(temp_df)
        sdf = pd.concat(split_dfs)
        out_df = df.copy()
        out_df = out_df[out_df['patient_id'].isin(sdf['patient_id'])]
        out_df.to_csv(split_fp)
    with open(dict_fp, "w") as fp:
        json.dump(overall_dict, fp, indent=2)


if __name__ == '__main__':
    for R in range(2,3):
        validation_file = f"/scratch/alexis.burgon/2022_CXR/model_runs/Monte_Carlo_and_Calibrate/attempt_1/RAND_{R}/validation.csv"
        divide_validation(input_file=validation_file)
    print("DONE")