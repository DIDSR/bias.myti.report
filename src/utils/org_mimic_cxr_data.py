'''
    Program to organize mimic-cxr data to be used for pretraining
    Creates json summary file

    Header for cxr-study-list.csv: subject_id,study_id,path
    Header for cxr-record-list.csv: subject_id,study_id,dicom_id,path

    subject_id: patient unique name
    study_id: each subjet can have many studies
    cxr-study-list/path: path to the 
    cxr-record-list: path to the dicom file

    Example script:
    python org_mimic_cxr_data.py -i /gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/ -r /gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv -s /gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv -o /gpfs_projects/ravi.samala/OUT/mimic/ -p /gpfs_projects/ravi.samala/DATA/mimic-iv/physionet.org/files/mimiciv/2.0/hosp/patients.csv.gz -a /gpfs_projects/ravi.samala/DATA/mimic-iv/physionet.org/files/mimiciv/2.0/hosp/admissions.csv.gz -n /gpfs_projects/ravi.samala/DATA/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-negbio.csv.gz -m /gpfs_projects/ravi.samala/DATA/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz
'''
import argparse
import pydicom
import pandas as pd
import os
import datetime
from multiprocessing import Pool, Manager
from functools import partial
# import tqdm
import time
import sys
import datetime
pd.options.mode.chained_assignment = None  # default='warn'

manufacturer_lookup_table = {
	'Not Reported':['MISSING', 'Missing', ''],
	'Carestream':['CARESTREAM HEALTH', 'CARESTREAM', 'Carestream Health'],
	'Philips':['Philips Medical Systems'],
	'Fujifilm':['FUJIFILM Corporation'],
	'Konica Minolta':['KONICA MINOLTA'],
	'GE Healthcare':['"GE Healthcare"','GE MEDICAL SYSTEMS'],
	'Canon Inc.':['Canon Inc.'],
	'Riverain Technologies':['Riverain Technologies'],
	'Siemens':['SIEMENS'],
	'Samsung Electronics':['Samsung Electronics'],
	'Kodak':['KODAK'],
	'Iray':['IRAY'],
	'ACME': ['ACME'],
	'Agfa': ['Agfa'],
	'Konica Minolta': ['KONICA MINOLTA'],
	'Rayence': ['Rayence'],
	'Toshiba': ['Toshiba'],
	}
img_info_dict = {
		'modality':(0x0008,0x0060),
		'body part examined':(0x0018,0x0015),
		# 'view position':(0x0018,0x5101),
		'study date':(0x0008,0x0020),
		'manufacturer':(0x0008,0x0070),
		'manufacturer model name':(0x0008,0x1090)}
race_lookup_table = {
	'American Indian or Alaska Native':['AMERICAN INDIAN OR ALASKA NATIVE', 'AMERICAN INDIAN/ALASKA NATIVE'],
	'Asian':['ASIAN'],
    'Asian - Chinese': ['ASIAN - CHINESE'],
    'Asian - Korean': ['ASIAN - KOREAN'],
    'Asian - Asian Indian': ['ASIAN - ASIAN INDIAN'],
    'Asian - South East Asian': ['ASIAN - SOUTH EAST ASIAN'],
	'Black or African American': ['BLACK OR AFRICAN AMERICAN', 'BLACK/AFRICAN AMERICAN'],
    'Black - African': ['BLACK/AFRICAN'],
    'Black - Cape Verdean': ['BLACK/CAPE VERDEAN'],
    'Black - Caribbean island': ['BLACK/CARIBBEAN ISLAND'],
	'Native Hawaiian or other Pacific Islander':['NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'],
	'Not Reported': ['nan', 'MISSING', 'Missing', 'UNABLE TO OBTAIN', 'UNKNOWN', 'PATIENT DECLINED TO ANSWER'],
	'Other':['OTHER'],
	'White':['WHITE'],
    'White - Brazilian':['WHITE - BRAZILIAN'],
    'White - Other Eurpopean':['WHITE - OTHER EUROPEAN'],
    'White - Russian':['WHITE - RUSSIAN'],
    'White - Eastern European':['WHITE - EASTERN EUROPEAN'],
    'Multiple':['MULTIPLE RACE/ETHNICITY'],
    'Portuguese': ['PORTUGUESE'],
    'South American': ['SOUTH AMERICAN'],
    'Hispanic/Latino': ['HISPANIC OR LATINO'],
    'Hispanic/Latino - Dominican': ['HISPANIC/LATINO - DOMINICAN'],
    'Hispanic/Latino - Salvadoran': ['HISPANIC/LATINO - SALVADORAN'],
    'Hispanic/Latino - Puerto Rican': ['HISPANIC/LATINO - PUERTO RICAN' ],
    'Hispanic/Latino - Guatemalan': ['HISPANIC/LATINO - GUATEMALAN'],
    'Hispanic/Latino - Central American': ['HISPANIC/LATINO - CENTRAL AMERICAN' ],
    'Hispanic/Latino - Mexican': ['HISPANIC/LATINO - MEXICAN'],
    'Hispanic/Latino - Cuban': ['HISPANIC/LATINO - CUBAN' ],
    'Hispanic/Latino - Honduran': ['HISPANIC/LATINO - HONDURAN'],
    'Hispanic/Latino - Columbian': ['HISPANIC/LATINO - COLUMBIAN' ],
    }


def manufacturer_lookup(manufacturer_info):
	manufacturer_info = str(manufacturer_info)
	if manufacturer_info in manufacturer_lookup_table:
		return manufacturer_info
	for key in manufacturer_lookup_table:
		if manufacturer_info in manufacturer_lookup_table[key]:
			return key
	print(f"manufacturer value {manufacturer_info} not in lookup table")
	return manufacturer_info


def race_lookup(race_info):
	race_info = str(race_info)
	if race_info in race_lookup_table:
		return race_info
	for key in race_lookup_table:
		if race_info in race_lookup_table[key]:
			return key
	print(f"race value {race_info} not in lookup table")
	return race_info


def org_mimic_repo(args):
    # #
    # # read the cxr-record-list.csv to get the patient_id, study_id and the path to the dcm
    # # 
    record_pd = pd.read_csv(args.record_lis)
    print('There are {} rows in {}'.format(len(record_pd.index), args.record_lis))
    # # get the patients id info
    patient_ids = record_pd['subject_id'].unique()
    print('There are {} patient IDs in {}'.format(len(patient_ids), args.record_lis))
    df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
    out_summ_file = os.path.join(args.output_dir, 'mimic_cxr.json')
    pre, ext = os.path.splitext(out_summ_file)
    out_log_file = pre + '.log'
    start_time = datetime.datetime.now()
    with open(out_log_file, 'w') as fp:
        fp.write('START: ' + str(start_time) + '\n')
    num_patients_to_json = 0
    num_images_to_json = 0
    num_series_to_json = 0
    num_missing_images = 0
    # # iterate over each patient and gather necessary info
    for each_patient in patient_ids:
        if num_patients_to_json > 0 and num_patients_to_json % 50000 == 0:
            print('Processed {} patients with {} series and {} images so far...'.format(num_patients_to_json, num_series_to_json, num_images_to_json))
        imgs_good = []
        imgs_good_info = []
        imgs_bad = []
        imgs_bad_info = []
        patient_good_info = []
        # # get all the study_id rows match this patient
        patient_df = record_pd.loc[record_pd['subject_id'] == each_patient]
        for idx, each_study_id in patient_df.iterrows():
            num_images_to_json += 1
            num_series_to_json += 1
            if not os.path.exists(os.path.join(args.input_dir, each_study_id['path'])):
                num_missing_images += 1
            ds = pydicom.read_file(os.path.join(args.input_dir, each_study_id['path']))
            imgs_good.append(os.path.join(args.input_dir, each_study_id['path']))
            # get img info:
            img_info = {key: ds[img_info_dict[key][0],img_info_dict[key][1]].value if img_info_dict[key] in ds else 'MISSING' for key in img_info_dict}
            img_info['pixel spacing'] = [ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else 'MISSING'
            img_info['image size'] = ds.pixel_array.shape
            img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
            img_info['study_id'] = each_study_id['study_id']
            imgs_good_info.append(img_info)
        df.loc[len(df)] = [each_patient] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['mimic_cxr']
        num_patients_to_json += 1
        # if num_patients_to_json == 10:
        #     break
    # #
    # # save info
    print('Output summary file: {}'.format(out_summ_file))
    print('Saving {} patients to json'.format(num_patients_to_json))
    print('Saving {} images to json'.format(num_images_to_json))
    print('Saving {} series to json'.format(num_series_to_json))
    print('Missing {} images to json'.format(num_missing_images))
    df.to_json(out_summ_file, indent=4, orient='table', index=False)
    print('Log file saved at: ' + out_log_file)
    print('json file saved at: ' + out_summ_file)
    end_time = datetime.datetime.now()
    with open(out_log_file, 'a') as fp:
        fp.write('END: ' + str(end_time) + '\n')
        fp.write('It took: ' +  str(end_time - start_time) + '\n')
        fp.write('Saved {} patients in json\n'.format(num_patients_to_json))
        fp.write('Saved {} images in json\n'.format(num_images_to_json))
        fp.write('Saved {} series in json\n'.format(num_series_to_json))
        fp.write('Missed {} images in json\n'.format(num_missing_images))


def preprocess_mimic(args):
    '''
        From: https://github.com/MLforHealth/CXR_Fairness/
    '''
    patients = pd.read_csv(args.patients_csv)
    # print(list(patients.columns))
    # temp = pd.read_csv(args.admissions_csv)
    # print(list(temp.columns))
    # print(temp.race.unique())
    # ethnicities = pd.read_csv(args.admissions_csv).drop_duplicates(subset = ['subject_id']).set_index('subject_id')['ethnicity'].to_dict()
    # patients['ethnicity'] = patients['subject_id'].map(ethnicities)
    race = pd.read_csv(args.admissions_csv).drop_duplicates(subset = ['subject_id']).set_index('subject_id')['race'].to_dict()
    insurance = pd.read_csv(args.admissions_csv).drop_duplicates(subset = ['subject_id']).set_index('subject_id')['insurance'].to_dict()
    patients['race'] = patients['subject_id'].map(race)
    patients['insurance'] = patients['subject_id'].map(insurance)
    labels = pd.read_csv(args.negbio_csv)
    meta = pd.read_csv(args.metadata_csv)

    df = meta.merge(patients, on = 'subject_id').merge(labels, on = ['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins = list(range(0, 101, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['path'] = df.apply(lambda x: os.path.join('files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'), axis = 1)
    
    # df = split(df.reset_index(drop = True))
    now = datetime.datetime.now()
    df.to_csv(os.path.join(args.output_dir, '{}{:02d}{:02d}'.format(now.year, now.month, now.day)  + '_preprocessed__mimic_cxr.csv'), index=False)
    return df


def read_patient_data_parallel(lst, patient_df2):
    df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
    patient_df = patient_df2[0]
    imgs_good = []
    imgs_good_info = []
    imgs_bad = []
    imgs_bad_info = []
    patient_good_info = []
    # # create patient-level info
    each_patient = 'ERROR'
    for idx, each_study_id in patient_df.iterrows():
        each_patient = each_study_id['subject_id']
        ds = pydicom.read_file(each_study_id['path'])
        imgs_good.append(each_study_id['path'])
        # # get img info:
        img_info = {key: ds[img_info_dict[key][0],img_info_dict[key][1]].value if img_info_dict[key] in ds else 'MISSING' for key in img_info_dict}
        img_info['pixel spacing'] = [ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else 'MISSING'
        img_info['image size'] = ds.pixel_array.shape
        img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
        img_info['study_id'] = each_study_id['study_id']
        img_info['study_time'] = each_study_id['StudyTime']
        # # based on the CR or DX, select the approporate DCM tag for view
        if 'DX' in ds[0x0008,0x0060].value:
            if (0x0018,0x5101) in ds:
                img_info['view position'] = ds[0x0018,0x5101].value
            else:
                img_info['view position'] = 'UNKNOWN DX VIEW POSITION'
        elif 'CR' in ds[0x0008,0x0060].value:
            if (0x0018,0x1400) in ds:
                img_info['view position'] = ds[0x0018,0x1400].value
            else:
                img_info['view position'] = 'UNKNOWN CR VIEW POSITION'
        else:
            img_info['view position'] = 'ERROR: UNKNOWN: ' + ds[0x0008,0x0060].value
        imgs_good_info.append(img_info)
        # #
        patient_info = {
                    'sex': each_study_id['gender'],
                    'race': race_lookup(each_study_id['race']),
                    'ethnicity': "Not available",
                    'COVID_positive': "Not applicable",
                    'age': each_study_id['age_decile'],
                    # 'insurance': each_study_id['insurance'] if each_study_id['insurance'] and not each_study_id['insurance'].isspace() else 'Missing',
                    'insurance': each_study_id['insurance'],
                    'frontal': each_study_id['frontal'],
                    # # diseases
                    'Atelectasis': each_study_id['Atelectasis'],
                    'Cardiomegaly': each_study_id['Cardiomegaly'],
                    'Consolidation': each_study_id['Consolidation'],
                    'Edema': each_study_id['Edema'],
                    'Enlarged Cardiomediastinum': each_study_id['Enlarged Cardiomediastinum'],
                    'Fracture': each_study_id['Fracture'],
                    'Lung Lesion': each_study_id['Lung Lesion'],
                    'Lung Opacity': each_study_id['Lung Opacity'],
                    'No Finding': each_study_id['No Finding'],
                    'Pleural Effusion': each_study_id['Pleural Effusion'],
                    'Pleural Other': each_study_id['Pleural Other'],
                    'Pneumonia': each_study_id['Pneumonia'],
                    'Pneumothorax': each_study_id['Pneumothorax'],
                    }
    patient_good_info = [patient_info]
    df.loc[0] = [each_patient] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['mimic_cxr']
    lst.append(df)
    return 0

def track_job(job, update_interval=3, total_jobs=100):
    while job._number_left > 0:
        # print("Tasks remaining = {0}".format(job._number_left * job._chunksize))
        sys.stderr.write('\rTasks remaining =  {:05d}/{}'.format(job._number_left * job._chunksize, total_jobs))
        time.sleep(update_interval)


def org_mimic_repo_parallel(args):
    # #
    # # read the cxr-record-list.csv to get the patient_id, study_id and the path to the dcm
    # # 
    # # 
    patient_processed_df = preprocess_mimic(args)
    record_pd_master = pd.read_csv(args.record_lis)
    # # get the patient info
    # # match the record_pd with the patient_info_df based on the "dicom_id" column
    print('cxr-record-list has {} rows'.format(len(record_pd_master.index)))
    print('patient-processed-list has {} rows'.format(len(patient_processed_df.index)))
    record_pd = record_pd_master.merge(patient_processed_df, left_on='dicom_id', right_on='dicom_id', suffixes=('', '_proc'))
    print('master-list has {} rows'.format(len(record_pd.index)))
    # # split the entire data based on the partition_dir
    record_pd['path'] = args.input_dir + '/' + record_pd['path'].astype(str)
    

    print('There are {} rows in processed list file'.format(len(record_pd.index)))
    # # get the patients id info
    patient_ids = record_pd['subject_id'].unique()
    print('There are {} patient IDs in processed list file'.format(len(patient_ids)))
    df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
    # out_summ_file = os.path.join(args.output_dir, partition_string + '__mimic_cxr.json')
    now = datetime.datetime.now()
    out_summ_file = os.path.join(args.output_dir, '{}{:02d}{:02d}'.format(now.year, now.month, now.day)  + '_summary_table__mimic_cxr.json')
    pre, ext = os.path.splitext(out_summ_file)
    out_log_file = pre + '.log'
    start_time = datetime.datetime.now()
    with open(out_log_file, 'w') as fp:
        fp.write('START: ' + str(start_time) + '\n')
    num_patients_to_json = 0
    dfs_list = Manager().list()
    patient_df_lst = list()
    # # iterate over each patient and gather necessary info
    for each_patient in patient_ids:
        # # get all the study_id rows match this patient
        patient_df = record_pd.loc[record_pd['subject_id'] == each_patient]
        patient_df_lst.append([patient_df])
        num_patients_to_json += 1
        # if num_patients_to_json == 100:
        #     break
    # # start parallel
    print('Saving {} patients to json'.format(num_patients_to_json))
    pool = Pool(os.cpu_count()-1)
    res = pool.map_async(partial(read_patient_data_parallel, dfs_list), patient_df_lst)
    # res.wait()
    track_job(res, 3, num_patients_to_json)
    dfs = pd.concat(dfs_list, ignore_index=True)  # the final result
    # # #
    # # # save info
    print('Output summary file: {}'.format(out_summ_file))
    print('Saved {} patients to json'.format(num_patients_to_json))
    dfs.to_json(out_summ_file, indent=4, orient='table', index=False)
    print('Log file saved at: ' + out_log_file)
    print('json file saved at: ' + out_summ_file)
    end_time = datetime.datetime.now()
    with open(out_log_file, 'a') as fp:
        fp.write('END: ' + str(end_time) + '\n')
        fp.write('It took: ' +  str(end_time - start_time) + '\n')
        fp.write('Saved {} patients in json\n'.format(num_patients_to_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='organize mimic-cxr data')
    parser.add_argument('-i', '--input_dir', help='<Required> root dir for mimic-cxr', 
                        required=True, default='/gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/')
    parser.add_argument('-r', '--record_lis', help='<Required> records list file', 
                        required=True, default='/gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv')
    parser.add_argument('-s', '--study_list', help='<Required> study list file', 
                        required=True, default='/gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv')
    parser.add_argument('-o', '--output_dir', help='<Required> output dir to save list files', 
                        required=True, default='/gpfs_projects/ravi.samala/OUT/mimic/')
    # # #
    parser.add_argument('-p', '--patients_csv', help='<Required> path to patients.csv.gz', 
                        required=True, default='/gpfs_projects/ravi.samala/DATA/mimic-iv/physionet.org/files/mimiciv/2.0/hosp/patients.csv.gz')
    parser.add_argument('-a', '--admissions_csv', help='<Required> path to admissions.csv.gz', 
                        required=True, default='/gpfs_projects/ravi.samala/DATA/mimic-iv/physionet.org/files/mimiciv/2.0/hosp/admissions.csv.gz')
    parser.add_argument('-n', '--negbio_csv', help='<Required> path to mimic-cxr-2.0.0-negbio.csv.gz', 
                        required=True, default='/gpfs_projects/ravi.samala/DATA/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-negbio.csv.gz')
    parser.add_argument('-m', '--metadata_csv', help='<Required> path to mimic-cxr-2.0.0-metadata.csv.gz', 
                        required=True, default='/gpfs_projects/ravi.samala/DATA/mimic_cxr_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz')
    args = parser.parse_args()
    # #
    org_mimic_repo_parallel(args)
    print('DONE')
