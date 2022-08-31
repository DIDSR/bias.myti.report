import os
import pandas as pd
import sys
sys.path.insert(0,'..')
from main_summarize import race_lookup, ethnicity_lookup, manufacturer_lookup
'''
Allows summary to be updated with respect to:
    changes in race/ethnicity/manufacturer lookup values
    manually excluded bad files
without requiring the generation of a new summary file
'''
def update_summary(input_summary_file, bad_img_file, output_file = None):
    print("updating...")
    df = pd.read_json(input_summary_file, orient='table')
    if not output_file:
        # save backup
        df.to_json(input_summary_file.replace(".json","_backup.json"), indent=4, orient='table')
    # read in bad files
    with open(bad_img_file, 'r') as infile:
        bad_files = infile.read().split('\n')
    for ii, row in df.iterrows():
        print(f" {ii+1}/{len(df)}")
        for i, img in enumerate(row['images']):
            if img in bad_files:
                print("moved an image to bad_images")
                row['bad_images'].append(img)
                row['bad_images_info'].append(row['images_info'][i])
                row['images'].remove(img)
                row['images_info'].remove(row['images_info'][i])
    # save summary
    print('saving summary...')
    if output_file:
        df.to_json(output_file, indent=4, orient='table')
    else:
        df.to_json(input_summary_file, indent=4, orient='table')
    print('DONE')

def update_terminology(input_summary_file, output_file=None):
    print("updating...")
    df = pd.read_json(input_summary_file, orient='table')
    if not output_file:
        # save backup
        df.to_json(input_summary_file.replace(".json","_backup.json"), indent=4, orient='table')
    for ii, row in df.iterrows():
        sys.stdout.write(f"\r{ii+1}/{len(df)} patients reviewed")
        row['patient_info'][0]['race'] = race_lookup(row['patient_info'][0]['race'])
        row['patient_info'][0]['ethnicity'] = ethnicity_lookup(row['patient_info'][0]['ethnicity'])
        for i, img_info in enumerate(row['images_info']):
            row['images_info'][i]['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
    '''
    if output_file:
        df.to_json(output_file, indent=4, orient='table')
    else:
        df.to_json(input_summary_file, indent=4, orient='table')'''
    print("\ndone")
 

if __name__ == '__main__':
    #update_summary("/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/summary_table__open_AI.json", "/home/alexis.burgon/covid/data/open_AI_manually_deleted_images.txt")
    update_terminology("/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__MIDRC_RICORD_1C.json")
    