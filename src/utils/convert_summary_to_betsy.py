import os
import pandas as pd

original_path = "/gpfs_projects/"
betsy_path = "/projects01/didsr-aiml/"

def convert_to_betsy(input_summary, out_path):
    df = pd.read_json(input_summary, orient='table')
    for ii, row in df.iterrows():
        for i, img in enumerate(row['images']):
            row['images'][i] = img.replace(original_path, betsy_path)
        for i, img in enumerate(row['bad_images']):
            row['bad_images'][i] = img.replace(original_path, betsy_path)
    df.to_json(out_path, orient='table')
    print("DONE")

if __name__ == '__main__':
    convert_to_betsy("/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_AR.json",
                     '/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/summary_table__COVID_19_AR_betsy.json')