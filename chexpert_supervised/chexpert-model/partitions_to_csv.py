import os
import pandas as pd
import json

partition_base_dir = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/20220901"
partition_folders = [f"RAND_{i}_OPTION_1" for i in range(5)]

orig_summary_folder = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823"
tasks = {
    'patient_sex' :['M', 'F'],
    'image_modality':['CR', 'DX']
}
repo_name_adj = {
    "RICORD-1c":"MIDRC_RICORD_1C"
}

def convert_partitions():
    print()
    # set up columns based on tasks
    cols = ['Path']
    for i,j in tasks.items():
        for k in j:
            cols.append(k)
    # get jpeg files
    jpeg_folders = [fp for fp in os.listdir(orig_summary_folder) if os.path.isdir(os.path.join(orig_summary_folder, fp))]
    conv_tables = {}
    for fp in jpeg_folders:
        repo = fp.replace("_jpeg",'')
        with open(os.path.join(orig_summary_folder, fp, "conversion_table.json"),'r') as f:
            conv_tables[repo] = json.load(f)
    for partition in partition_folders:
        print(f"Working on {partition}")
        part_path = os.path.join(partition_base_dir, partition)
        if not os.path.exists(part_path):
            raise Exception(f"{part_path} is not a valid path")
        
        for pt in ['tr','ts']:
            out_df = pd.DataFrame(columns=cols)
            for file in os.listdir(part_path):
                if file.startswith(pt) and file.endswith('.json'):
                    infile = os.path.join(part_path, file)
            in_df = pd.read_json(infile, orient='table')
            for ii, row in in_df.iterrows():
                for key, val in conv_tables.items():
                    if repo_name_adj[row['repo']] in key:
                        jpeg_dict = val
                if type(row['images']) != list:
                    row['images'] = [row['images']]
                    row['images_info'] = [row['images_info']]
                
                for i, img in enumerate(row['images']):
                    info = {col:0 for col in cols}
                    # get jpeg path from dicom path
                    if img not in list(jpeg_dict['dicom'].values()):
                        print(f"bad img {img}")
                        continue
                    id = list(jpeg_dict['dicom'].values()).index(img)
                    img_idx = list(jpeg_dict['dicom'].keys())[id]
                    info["Path"] = jpeg_dict['jpeg'][img_idx]
                    for cls, sub in tasks.items():
                        n1, n2 = cls.split('_')
                        if n1 == 'patient':
                            if row['patient_info'][0][n2] not in sub:
                                print("bad val")
                            info[row['patient_info'][0][n2]] = 1
                        elif n1 == 'image':
                            if row['images_info'][i][n2] not in sub:
                                print("bad val")
                            info[row['images_info'][i][n2]] = 1
                    # add to df
                    if out_df.empty:
                        idx = 0
                    else:
                        idx = len(out_df)
                    out_df.loc[idx] = [info[val] for val in info]
            out_df.to_csv(infile.replace('.json','.csv'))

if __name__ == '__main__':
    convert_partitions()