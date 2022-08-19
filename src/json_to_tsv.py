from ast import arg
import os
import pandas as pd
import argparse


def get_repo(args):
    if args.betsy:
        summary_json = f"/scratch/alexis.burgon/2022_CXR/data_summarization/summary_table__{args.repo}.json"
        img_save_loc = f"/scratch/alexis.burgon/2022_CXR/CXR_jpegs/{args.repo}"
        tsv_save_loc = f"/scratch/alexis.burgon/2022_CXR/data_summarization/summary_table__{args.repo}.tsv"
    else:
        summary_json = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/summary_table__{args.repo}.json"
        img_save_loc = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/jpeg_testing{args.repo}"
        tsv_save_loc = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/summary_table__{args.repo}.tsv"
    return summary_json, img_save_loc, tsv_save_loc

def json_to_tsv(args):
    '''
    converts the information from the summary_json file and the dicom to jpeg conversion table into
    a tsv file to be used for generating partitions
    '''
    if args.stop_at:
        print(f"stopping at {args.stop_at} patients")
    in_df = pd.read_json(args.input, orient='table')
    conversion_table = os.path.join(args.jpeg_save_loc,'conversion_table.json')
    conversion_df = pd.read_json(conversion_table)
    out_df = pd.DataFrame(columns=['patient_id', 'dicom_file', 'Path', 'CR', 'DX', 'Female', 'Male'])
    for i, row in in_df.iterrows():
        if args.stop_at and i >= args.stop_at:
            break
        patient_id = row['patient_id']
        patient_sex = row['patient_info'][0]['sex']
        if patient_sex == 'F':
            female = 1
            male = 0
        elif patient_sex == 'M':
            male = 1
            female = 0
        else:
            print(f'unknown patient sex {patient_sex}')
            return
        for ii, img in enumerate(row['images']):
            print(conversion_df[conversion_df['dicom'] == img]['jpeg'])
            
            #jpeg_file = conversion_df[conversion_df['dicom'] == img]['jpeg']
            #print(type(jpeg_file))
            
            jpeg_file = conversion_df[conversion_df['dicom'] == img]['jpeg'].head(1).item()
            if not os.path.exists(jpeg_file):
                #print("jpeg file doesn't exists yet! be sure to convert first")
                #return
                continue
            # get img information
            #print(row['images_info'])
            img_mod = row['images_info'][0]['modality']
            if img_mod == 'CR':
                CR = 1
                DX = 0
            elif img_mod == 'DX':
                CR = 0
                DX = 1
            else:
                print(f"unknown modality {img_mod}")
                return
            # add to out_df
            if out_df.empty:
                idx = 0
            else:
                idx = out_df.index.max()+1
            out_df.loc[idx] = [patient_id, img, jpeg_file, CR, DX, female, male]
            
    out_df.to_csv(args.output, sep="\t")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser
    parser.add_argument('-i','--input',action='append', default=[],required=True)
    parser.add_argument('t','--tsv_output',action='append', default=[],required=True)
    parser.add_argument('-j','--jpeg_save_locs',action='append', default=[], required=True)
    parser.add_argument('-s','--stop_at', default=None)
    parser.add_argument('-b','--betsy', default=False)
    json_to_tsv(parser.parse_args())