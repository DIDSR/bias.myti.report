import  pandas as pd
import os

def check_portable(x):
    if str(x) == 'Not Reported':
        return 'Not Reported'
    elif 'port' in str(x).lower():
        return 'portable'
    else:
        return 'Non-portable'

def get_img_info(conv_table_fp, study_info_fp):
    conv_df = pd.read_json(conv_table_fp)
    info_df = pd.read_csv(study_info_fp, sep='\t')
    # get study id from dicom filepaths, use to map study description
    common_path = os.path.commonpath(list(conv_df['dicom'].values))
    conv_df['study_id'] = conv_df['dicom'].replace({common_path:''}, regex=True).apply( lambda x: x.split("/")[2])
    conv_df['study_description'] = conv_df['study_id'].map(info_df.set_index('study_uid')['study_description'])
    conv_df['study_description'].fillna("Not Reported", inplace=True)
    conv_df['portable'] = conv_df['study_description'].apply(check_portable)
    return conv_df

def get_portable(in_csv, conv_df=pd.read_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/portable_or_nonportable/20230302_open_A1.csv")):
    if type(in_csv) == str:
        print(in_csv)
        df = pd.read_csv(in_csv)
    else:
        df = in_csv.copy()
    df['portable'] = df['Path'].map(conv_df.set_index('jpeg')['portable'])
    return df

if __name__ == "__main__":
    df = get_img_info("/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_open_A1_jpegs/conversion_table.json", "/gpfs_projects/ravi.samala/DATA/MIDRC3/20221010_open_A1_all_Imaging_Studies.tsv")
    df.to_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/portable_or_nonportable/20230302_open_A1.csv")
    print()
