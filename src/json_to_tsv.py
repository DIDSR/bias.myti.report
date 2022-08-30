from ast import arg
import os
import pandas as pd
import argparse
import sys

out_classes = ['CR', 'DX', 'F', 'M']
#out_classes = ['CR', 'DX', 'F', 'M', 'Asian', 'Black or African American', 'White']
by_patient = {'sex':['F', 'M'],
    'race':['American Indian or Alaska Native', 'Asian', 'Black or African American',
            'Native Hawaiian or other Pacific Islander', 'Not Reported', 'Other', 'White']}
by_image = {'modality':['CR', 'DX']}

def get_repo(args):
    if args.betsy:
        summary_json = f"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/summary_table__{args.repo}.json"
        img_save_loc = f"/scratch/alexis.burgon/2022_CXR/CXR_jpegs/{args.repo}"
        tsv_save_loc = f"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/summary_table__{args.repo}.tsv"
    else:
        summary_json = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__{args.repo}.json"
        img_save_loc = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/{args.repo}_jpegs"
        tsv_save_loc = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__{args.repo}.tsv"
    return summary_json, img_save_loc, tsv_save_loc

def get_attributes(df_info, corr_dict):
    out_dict = {}
    for out_class in out_classes:
        for att in corr_dict:
            if out_class in corr_dict[att]:
                out_dict[out_class] = 0
                if df_info[att] == out_class:
                    out_dict[out_class] += 1
    return out_dict

def json_to_tsv(args):
    summary_json, img_save_loc, tsv_save_loc = get_repo(args)
    in_df = pd.read_json(summary_json, orient='table')
    cols = ['patient id', 'dicom file', 'Path'] + out_classes
    conversion_table = os.path.join(img_save_loc,'conversion_table.json')
    conversion_df = pd.read_json(conversion_table)
    out_df = pd.DataFrame(columns=cols)
    for i, row in in_df.iterrows():
        sys.stdout.write(f"\r{i+1}/{len(in_df)} patients complete")
        # information by patient
        patient_id = row['patient_id']
        patient_info = get_attributes(row['patient_info'][0], by_patient)
        for ii, img in enumerate(row['images']):
            if ii >= len(row['images_info']):
                continue
            # information by image
            dicom_file = img
            jpeg_path = conversion_df[conversion_df['dicom'] == img]['jpeg'].values[0]
            img_info = get_attributes(row['images_info'][ii], by_image)
            # add to df
            if out_df.empty:
                idx = 0
            else:
                idx = len(out_df) 
            info_list = []
            for out_class in out_classes:
                if out_class in patient_info:
                    info_list.append(patient_info[out_class])
                elif out_class in img_info:
                    info_list.append(img_info[out_class])
            out_df.loc[idx] = [patient_id] + [dicom_file] + [jpeg_path] + info_list
    out_df.to_csv(tsv_save_loc, sep="\t")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-i','--input',action='append', default=[],required=True)
    #parser.add_argument('-t','--tsv_output',action='append', default=[],required=True)
    #parser.add_argument('-j','--jpeg_save_locs',action='append', default=[], required=True)
    #parser.add_argument('-s','--stop_at', default=None)
    parser.add_argument('-b','--betsy', default=False)
    parser.add_argument('-r','--repo',required=True)
    json_to_tsv(parser.parse_args())
    print("\nDONE\n")
    #get_attributes(parser.parse_args())