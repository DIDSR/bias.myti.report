#from portable_or_nonportable import get_portable, summarize_portable
from argparse import ArgumentParser
import pandas as pd
import os

def get_days_to_test(s_file="/gpfs_projects/ravi.samala/DATA/MIDRC3/20221010_open_A1_all_Imaging_Studies.tsv",
                     m_file="/gpfs_projects/ravi.samala/DATA/MIDRC3/20221010_open_A1_all_Measurements.tsv",
                     c_file="/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_open_A1_jpegs/conversion_table.json"):
    """
    Gets the days to test for each individual study. Currently only tested for open-A1
    """
    # NOTE: for open-A1, all tests are for COVID-19
    s_df = pd.read_csv(s_file,sep='\t')
    s_df = s_df[~s_df['days_to_study'].isna()]
    m_df = pd.read_csv(m_file,sep='\t')
    m_df = m_df[~m_df['test_days_from_index'].isna()]
    m_df = m_df.sort_values('case_ids_0')
    s_df = s_df.sort_values('case_ids_0')
    # positive tests
    pos_m = m_df[m_df['test_result_text'] == 'Positive'][['test_result_text', 'case_ids_0','test_days_from_index']]
    pos_df = pd.merge(s_df[s_df['case_ids_0'].isin(pos_m['case_ids_0'])][['study_uid','days_to_study','case_ids_0']], pos_m, on='case_ids_0')
    pos_df['days_from_study_to_positive_test'] = pos_df['test_days_from_index'] - pos_df['days_to_study']
    pos_df['abs_days'] = pos_df['days_from_study_to_positive_test'].abs()
    # print(pos_df.sort_values('abs_days')[['case_ids_0', 'abs_days']].head())
    pos_df = pos_df.sort_values('abs_days').groupby(['case_ids_0','study_uid']).apply(pd.DataFrame.head,n=1).reset_index(drop=True) # we only care about the test closest to the study date

    # pos_df = pos_df.groupby(['case_ids_0','study_uid']).min().reset_index() # we only care about the test closest to the study date
    # negative tests
    neg_m = m_df[m_df['test_result_text'] == 'Negative'][['test_result_text', 'case_ids_0','test_days_from_index']]
    neg_df = pd.merge(s_df[s_df['case_ids_0'].isin(neg_m['case_ids_0'])][['study_uid','days_to_study','case_ids_0']], neg_m, on='case_ids_0')
    neg_df['days_from_study_to_negative_test'] = neg_df['test_days_from_index'] - neg_df['days_to_study']
    neg_df['abs_days'] = neg_df['days_from_study_to_negative_test'].abs()
    neg_df = neg_df.sort_values('abs_days').groupby(['case_ids_0','study_uid']).apply(pd.DataFrame.head,n=1).reset_index(drop=True) # we only care about the test closest to the study date
    # neg_df = neg_df.groupby(['case_ids_0','study_uid']).min().reset_index() # we only care about the test closest to the study date
    # combine information
    df = s_df[['study_uid','days_to_study','case_ids_0']].copy()
    df['days_from_study_to_positive_test'] = df['study_uid'].map(pos_df.set_index(['study_uid'])['days_from_study_to_positive_test'])
    df['days_from_study_to_negative_test'] = df['study_uid'].map(neg_df.set_index(['study_uid'])['days_from_study_to_negative_test'])
    # map to individual images
    c_df = pd.read_json(c_file) # conversion table
    common_path = os.path.commonpath(list(c_df['dicom'].values))
    c_df['study_id'] = c_df['dicom'].replace({common_path:''}, regex=True).apply(lambda x: x.split("/")[2])
    c_df['days_from_study_to_positive_test'] = c_df['study_id'].map(df.set_index('study_uid')['days_from_study_to_positive_test'])
    c_df['days_from_study_to_negative_test'] = c_df['study_id'].map(df.set_index('study_uid')['days_from_study_to_negative_test'])
    return c_df

def img_days_to_test(input_csv, test_date_df, imgs_per_patient=1, allow_null=False, portable_info=False):
    """
    Uses the study-test information from get_days_to_test to get the days to (relevant) test for each image in a provided csv,
    limits the number of images per patient to imgs_per_patient based on how many days between image and test
    """
    test_df = test_date_df
    if type(input_csv) == str:
        df = pd.read_csv(input_csv)
    else:
        df = input_csv.copy()
    print(f"Loaded information for {len(df)} images ({df['patient_id'].nunique()} patients)")
    # positive cases
    pos_df = df[df['Yes'] == 1].copy()
    pos_df['days_from_study_to_test'] = pos_df['Path'].map(test_df.set_index('jpeg')['days_from_study_to_positive_test'])
    # negative cases
    neg_df = df[df['No'] == 1].copy()
    neg_df['days_from_study_to_test'] = neg_df['Path'].map(test_df.set_index('jpeg')['days_from_study_to_negative_test'])
    df = pd.concat([pos_df, neg_df], axis=0)
    if not allow_null:
        df = df[~df['days_from_study_to_test'].isnull()]
        print(f"Removing images without test date information, now have {len(df)} images ({df['patient_id'].nunique()} patients)")
    else:
        # to prioritize samples with test days, set those w/o test information as very far in the future
        df['days_from_study_to_test'] = df['days_from_study_to_test'].fillna(10000000)
    print(f"limiting to {imgs_per_patient} image(s) per patient...")
    out_df = df.copy()
    out_df['abs_days_from_study_to_test'] = out_df['days_from_study_to_test'].abs()
    out_df = out_df.sort_values('abs_days_from_study_to_test').groupby(['patient_id']).apply(pd.DataFrame.head,n=imgs_per_patient).reset_index(drop=True)
    print(f"Output dataset contains {len(out_df)} images for {out_df['patient_id'].nunique()} patients")
    out_df = out_df.drop(['abs_days_from_study_to_test'], axis=1)
    out_df = out_df.drop(['days_from_study_to_test'], axis=1)
    if portable_info:
        print("Getting portable/non-portable information for each patient")
        out_df = get_portable(out_df)
        summ_df = summarize_portable(out_df)
        return out_df, summ_df
    return out_df
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i','--input_csv', help='existing csv to be limited to a certain number of images per patient', required=True)
    parser.add_argument('-o','--output_csv', help='filepath to save limited csv to', required=True)
    parser.add_argument('-n', '--num_images', help='number of images to limit each patient to', default=True)
    parser.add_argument('--allow_null', action='store_true', default=False)
    parser.add_argument("--portable", action='store_true', default=False)
    args = parser.parse_args()
    # df = get_days_to_test()
    # print(df.head())
    # df.to_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/study_to_test_date/20230306_open_A1.csv")
    df = pd.read_csv("/scratch/yuhang.zhang/OUT/data_summarization/20230306_open_A1.csv")
    # df.to_csv("/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20221010_betsy/20230301_open_A1_test_info.csv")
    if args.portable:
        out_df, summ_df = img_days_to_test(args.input_csv, df, allow_null=args.allow_null, portable_info=args.portable)
        summ_df.to_csv(args.output_csv.replace('.csv','_summary.csv'), index=False)
    else:
        out_df = img_days_to_test(args.input_csv, df, allow_null=args.allow_null, portable_info=args.portable)
    out_df.to_csv(args.output_csv, index=False)

    
