'''
    Program to organize mimic-cxr data to be used for pretraining

    Header for cxr-study-list.csv: subject_id,study_id,path
    Header for cxr-record-list.csv: subject_id,study_id,dicom_id,path

    subject_id: patient unique name
    study_id: each subjet can have many studies
    cxr-study-list/path: path to the 
    cxr-record-list: path to the dicom file

    Example script:
    python org_mimic_cxr_data.py -i /gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/ -r /gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv -s /gpfs_projects/ravi.samala/DATA/mimic_cxr/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv -o /gpfs_projects/ravi.samala/OUT/mimic/
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
		'view position':(0x0018,0x5101),
		'study date':(0x0008,0x0020),
		'manufacturer':(0x0008,0x0070),
		'manufacturer model name':(0x0008,0x1090)}


def manufacturer_lookup(manufacturer_info):
	manufacturer_info = str(manufacturer_info)
	if manufacturer_info in manufacturer_lookup_table:
		return manufacturer_info
	for key in manufacturer_lookup_table:
		if manufacturer_info in manufacturer_lookup_table[key]:
			return key
	print(f"manufacturer value {manufacturer_info} not in lookup table")
	return manufacturer_info


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


def read_patient_data_parallel(lst, patient_df2):
    df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
    patient_df = patient_df2[0]
    imgs_good = []
    imgs_good_info = []
    imgs_bad = []
    imgs_bad_info = []
    patient_good_info = []       
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
        imgs_good_info.append(img_info)
    # lst.append([each_patient] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['mimic_cxr'])
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
    # partition_dir = ['files/p10/', 'files/p11/', 'files/p12/', 'files/p13/', 'files/p14/', 'files/p15/', 'files/p16/', 'files/p17', 'files/p18/', 'files/p19/']
    partition_dir = ['files/p10/', 'files/p11/', 'files/p12/']
    record_pd_master = pd.read_csv(args.record_lis)
    # # split the entire data based on the partition_dir
    for each_partition_dir in partition_dir:
        partition_string = each_partition_dir.replace('/', '_')
        print('=====================')
        print(partition_string)
        record_pd = record_pd_master[record_pd_master['path'].str.contains(each_partition_dir)]
        record_pd['path'] = args.input_dir + '/' + record_pd['path'].astype(str)
        print('There are {} rows in {}'.format(len(record_pd.index), each_partition_dir))
        # # get the patients id info
        patient_ids = record_pd['subject_id'].unique()
        print('There are {} patient IDs in {}'.format(len(patient_ids), each_partition_dir))
        df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
        out_summ_file = os.path.join(args.output_dir, partition_string + '__mimic_cxr.json')
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
            if num_patients_to_json == 100:
                break
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
    # parser.add_argument('-n', '--num_of_random_patients', 
    #                     help='Number of random patients to select for analysis', 
    #                     type=int, default=1000)
    args = parser.parse_args()
    # #
    org_mimic_repo_parallel(args)
    print('DONE')
