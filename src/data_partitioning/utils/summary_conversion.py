import pandas as pd
from constants import CXR_patient_info, CXR_image_info
import os

# General Functions =================================================================================================
def convert_summary_format(summary_filepath:str, conversion_table_filepath:str, attributes:list, repo:str, portable_fp:str):
    if repo == 'BraTS':
        df = convert_summary_format_BraTS(summary_filepath, attributes)
    else: 
        df = convert_summary_format_CXR(summary_filepath, conversion_table_filepath, attributes, repo, portable_fp)
    return df

def get_subgroup(row, attributes):
    """ gets the subgroup of a single row """
    subgroup = ""
    for a in attributes:
        if len(subgroup) == 0:
            subgroup = row[a]
        else:
            subgroup = subgroup + "-" +row[a]
    return subgroup

def load_existing_csvs(existing_csvs, args):
    partition_csvs = {f"Step {s}":{} for s in range(args.steps)}
    total_loaded = pd.DataFrame(columns=[args.id_col])
    for fp in existing_csvs:
        filename = fp.split("/")[-1]
        for s in range(args.steps):
            if args.steps == 1:
                partition_name = filename.split(".")[0]
                print(f"Loading {partition_name} partition from file...")
            else:
                partition_name = filename.split(".")[0].split("__")[-1]
                print(f"Loading Step {s} - {partition_name} partition from file...")
            in_df = pd.read_csv(fp)
            partition_csvs[f"Step {s}"][partition_name] = in_df[~in_df[args.id_col].isin(total_loaded[args.id_col])] # Undo effects of accumulate while loading
            total_loaded = pd.concat([total_loaded, partition_csvs[f"Step {s}"][partition_name]])
    return partition_csvs
# Segmentation ========================================================================================================
def convert_summary_format_BraTS(summary_filepath:str, attributes:list):
    """ Convert mapping csv format """
    df = pd.read_csv(summary_filepath)
    df['patient_id'] = df['BraTS2021']
    # get year added
    years = ["2021", "2020", "2019", "2018","2017"]
    df['year_added'] = df.notna().dot(df.columns+',').str.rstrip(",")
    df['year_added'] = df.apply(lambda row: get_year_added(row, years), axis=1)
    df.drop([f"BraTS{year}" for year in years], axis=1, inplace=True)
    df.rename({"Cohort Name (if publicly available)":"Cohort", "Site No (represents the originating institution)":'Site'}, inplace=True, axis=1)
    # get the filepaths to different scans
    image_folder = os.path.join(os.path.dirname(summary_filepath),"BraTS_data","RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021")
    scans = ['t2','t1','t1ce','flair']
    for scan in scans + ['seg']:
        df[scan] = df['patient_id'].apply(lambda p:  os.path.join(image_folder, str(p), f"{p}_{scan}.nii.gz"))
    df = df.melt(id_vars=[c for c in df.columns if c not in scans], value_vars=scans, var_name='scan_type', value_name='Path')
    df['subgroup'] = df.astype(str).apply(lambda row: get_subgroup(row, attributes), axis=1)
    return df.astype(str)

def get_year_added(row, years):
    l = row['year_added'].split(",")
    l = [int(x.replace("BraTS","")) for x in l if x.replace("BraTS","") in years]
    return min(l)

# Classification ======================================================================================================
def convert_summary_format_CXR(summary_filepath:str, conversion_table_filepath:str, attributes:list, repo:str, portable_fp:str, remove_lateral=True):
    """ Convert json format to csv-style format DataFrame """
    df = pd.read_json(summary_filepath, orient='table')
    # extract individual patient attribute information
    for a in CXR_patient_info:
        df[a] = df.apply(lambda row: row['patient_info'][0][a], axis=1)
    df = df.drop('patient_info', axis=1)
    # change from each row being one patient to each row being one image
    exp_cols = {'images', 'images_info'}
    other_cols = list(set(df.columns)-exp_cols)
    exploded = [df[col].explode() for col in exp_cols]
    temp_df = pd.DataFrame(dict(zip(exp_cols, exploded)))
    temp_df = df[other_cols].merge(temp_df, how='right', left_index=True, right_index=True)
    temp_df = temp_df.loc[:, df.columns] # get original column order
    df = temp_df.copy()
    # extract individual image attribute information
    for a in CXR_image_info:
        df[a] = df.apply(lambda row: row['images_info'][a.replace("_"," ")], axis=1)
    df = df.drop('images_info', axis=1)
    df = get_portable_nonportable(df,portable_fp)
    # convert dicom filepaths to jpeg filepaths
    conv_table = pd.read_json(conversion_table_filepath)
    df['Path'] = df['images'].map(conv_table.set_index('dicom')['jpeg'])
    df = df.drop('images', axis=1)
    # track source repository
    df['repo'] = repo
    # get subgroup information
    df['subgroup'] = df.apply(lambda row: get_subgroup(row, attributes), axis=1)
    if remove_lateral:# remove any lateral images
        good_vp = [x for x in df['view_position'].unique() if 'LAT' not in x]
        df = df[df['view_position'].isin(good_vp)].copy()
    
    return df

def get_portable_nonportable(df, portable_fp):
    """ Gets the portable/nonportable attribute for CXR """
    pdf = pd.read_csv(portable_fp, sep="\t" if portable_fp.endswith(".tsv") else ',')
    pdf = pdf[['study_uid', 'study_description']].copy()
    # get STUDY id for each image
    temp_df = df[['patient_id', 'images']].copy()
    cp = os.path.commonpath(temp_df['images'].values.tolist())
    temp_df['study_uid'] = temp_df['images'].replace({cp:""}, regex=True)
    temp_df['study_uid'] = temp_df.apply(lambda row: row['study_uid'].replace(f"/{row['patient_id']}/","").split("/")[0],axis=1)
    
    temp_df['study_description'] = temp_df['study_uid'].map(pdf.set_index("study_uid")['study_description'])
    temp_df['study_description'] = temp_df['study_description'].fillna("Not Reported")
    temp_df['portable'] = temp_df['study_description'].apply(check_portable)
    df['portable'] = df['images'].map(temp_df.set_index('images')['portable'])

    return df

def check_portable(x):
    if x == 'Not Reported':
        return 'Not Reported'
    elif 'port' in str(x).lower():
        return 'portable'
    else:
        return 'non-portable'



