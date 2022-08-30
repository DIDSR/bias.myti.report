import os
import pandas as pd
import argparse
'''
converts file paths to be betsy compatible
'''
original_path = "/gpfs_projects/"
betsy_path = "/projects01/didsr-aiml/"

def convert_to_betsy(input_summary, out_path):
    
    df = pd.read_json(input_summary, orient='table')
    for ii, row in df.iterrows():
        for i, img in enumerate(row['images']):
            row['images'][i] = img.replace(original_path, betsy_path)
        if 'bad images' in row:
            for i, img in enumerate(row['bad_images']):
                row['bad_images'][i] = img.replace(original_path, betsy_path)
    df.to_json(out_path, orient='table')
    print("DONE")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--repo',required=True)
    args = parser.parse_args()
    input_summary = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__{args.repo}.json"
    output_summary = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823_betsy/summary_table__{args.repo}.json"
    convert_to_betsy(input_summary, output_summary)