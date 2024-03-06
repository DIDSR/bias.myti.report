import os
import glob
import fnmatch
import argparse
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import datetime
from ipywidgets import IntProgress
from IPython.display import display
# #
# consistent terminology
race_lookup_table = {
	'American Indian or Alaska Native':['AMERICAN INDIAN OR ALASKA NATIVE'],
	'Asian':['ASIAN'],
	'Black or African American': ['BLACK OR AFRICAN AMERICAN'],
	'Native Hawaiian or other Pacific Islander':['NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'],
	'Not Reported': ['nan', 'MISSING', 'Missing'],
	'Other':['OTHER'],
	'White':['WHITE']}
ethnicity_lookup_table = {
	'Hispanic or Latino':[],
	'Not Hispanic or Latino':[],
	'Not Reported':['MISSING', 'Missing']}
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
modality_choices = ['CR', 'DX']

def race_lookup(race_info):
	race_info = str(race_info)
	if race_info in race_lookup_table:
		return race_info
	for key in race_lookup_table:
		if race_info in race_lookup_table[key]:
			return key
	print(f"race value {race_info} not in lookup table")
	return race_info

def ethnicity_lookup(ethnicity_info):
	ethnicity_info = str(ethnicity_info)
	if ethnicity_info in ethnicity_lookup_table:
		return ethnicity_info
	for key in ethnicity_lookup_table:
		if ethnicity_info in ethnicity_lookup_table[key]:
			return key
	print(f"ethnicity value {ethnicity_info} not in lookup table")
	return ethnicity_info

def manufacturer_lookup(manufacturer_info):
	manufacturer_info = str(manufacturer_info)
	if manufacturer_info in manufacturer_lookup_table:
		return manufacturer_info
	for key in manufacturer_lookup_table:
		if manufacturer_info in manufacturer_lookup_table[key]:
			return key
	print(f"manufacturer value {manufacturer_info} not in lookup table")
	return manufacturer_info

def get_dcms(file_path):
    dcms = []
    for p, d, f in os.walk(file_path):
        for file in f:
            if file.endswith('.dcm'):
                fp = os.path.join(p, file)
                dcms.append(fp)
    return dcms

def searchthis(location, searchterm):
	lis_paths = []
	for dir_path, dirs, file_names in os.walk(location):
		for file_name in file_names:
			fullpath = os.path.join(dir_path, file_name)
			if searchterm in fullpath:
				lis_paths += [fullpath]
	return lis_paths

	
def read_open_A1_20221010(args):
	'''
	using the imaging data and the associated MIDRC tsv files downloaded on 20221010

	Info on supporting files:
		../data/20221010_open_A1_all_Cases.tsv: get patient-level info (submitter_id, sex, age, race, COVID_status)
		../data/20221010_open_A1_all_Imaging_Studies.tsv: for a submitter_id (case_ids_0), use the study_modality_0
			identify the study_uid which is the subdirectory name. However, sometimes, the study_uid is the main directory
	'''
	# information to gather (pixel spacing and img size done separately)
	img_info_dict = {
	'modality':(0x0008,0x0060),
	'body part examined':(0x0018,0x0015),
	'view position':(0x0018,0x5101),
	'study date':(0x0008,0x0020),
	'manufacturer':(0x0008,0x0070),
	'manufacturer model name':(0x0008,0x1090)}
	# # get patient info
	patient_df = pd.read_csv(args.case_tsv, sep='\t')
	img_series_df = pd.read_csv(args.series_tsv, sep='\t')
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
	print('There are {} patients'.format(len(patient_df.index)))
	progress_bar = IntProgress(min=0, max=len(patient_df.index), description='Reading:')
	display(progress_bar)  
	# # iterate over the patient-id
	num_patients_to_json = 0
	num_images_to_json = 0
	num_series_to_json = 0
	num_missing_images = 0
	for idx, patient_row in patient_df.iterrows():
		progress_bar.value += 1
		if num_patients_to_json > 0 and num_patients_to_json % 1000 == 0:
				print('Processed {} patients with {} series and {} images so far...'.format(num_patients_to_json, num_series_to_json, num_images_to_json))
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		imgs_bad_info = []
		patient_skip = True
		# #
		patient_id = patient_row['submitter_id']
		# # identify the dir/sub-dir based on the Imaging_Studies file
		df_patient = img_series_df.loc[img_series_df['case_ids_0'] == patient_id]
		for study_idx, study_row in df_patient.iterrows():
			if study_row['modality'] in modality_choices:
				patient_study_path = None
				patient_study_path1 = os.path.join(args.input_dir, study_row['case_ids_0'], str(study_row['study_uid_0']), str(study_row['series_uid']))
				patient_study_path2 = os.path.join(args.input_dir, str(study_row['study_uid_0']), str(study_row['series_uid']))
				if os.path.exists(patient_study_path1):
					patient_study_path = patient_study_path1
				elif os.path.exists(patient_study_path2):
					patient_study_path = patient_study_path2
				else:
					num_missing_images += 1
				if patient_study_path is not None and study_row['modality'] in modality_choices:
					patient_skip = False
					# # get the dicom info here
					# # there should be at least 1 dicom file in this folder
					dcm_files = glob.glob(os.path.join(patient_study_path, '*.dcm'))
					num_images_to_json += len(dcm_files)
					num_series_to_json += 1
					for each_dcm in dcm_files:
						ds = pydicom.read_file(each_dcm)
						imgs_good.append(each_dcm)
						# get img info:
						img_info = {key: ds[img_info_dict[key][0],img_info_dict[key][1]].value if img_info_dict[key] in ds else 'MISSING' for key in img_info_dict}
						img_info['pixel spacing'] = [ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else 'MISSING'
						img_info['image size'] = ds.pixel_array.shape
						img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
						imgs_good_info.append(img_info)

		# # create patient-level info
		patient_info = {
						'sex':"M" if patient_row['sex'] == "Male" else "F" if patient_row['sex'] == "Female" else "Unknown",
						'race':race_lookup(patient_row['race']),
						'ethnicity':ethnicity_lookup(patient_row['ethnicity']),
						'COVID_positive':patient_row['covid19_positive'],
						'age':patient_row['age_at_index'],
						}
		patient_good_info = [patient_info]
		# add to df
		if not patient_skip:
			df.loc[len(df)] = [patient_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['open-A1']
			num_patients_to_json += 1
	# # print summary info and save output files
	print('Saving {} patients to json'.format(num_patients_to_json))
	print('Saving {} images to json'.format(num_images_to_json))
	print('Saving {} series to json'.format(num_series_to_json))
	print('Missing {} images to json'.format(num_missing_images))
	df.to_json(args.output_file, indent=4, orient='table', index=False)
	pre, ext = os.path.splitext(args.output_file)
	out_log_file = pre + '.log'
	print('Log file saved at: ' + out_log_file)
	print('json file saved at: ' + args.output_file)
	with open(out_log_file, 'w') as fp:
		fp.write(str(datetime.datetime.now()) + '\n')
		fp.write('Saved {} patients in json\n'.format(num_patients_to_json))
		fp.write('Saved {} images in json\n'.format(num_images_to_json))
		fp.write('Saved {} series in json\n'.format(num_series_to_json))
		fp.write('Missed {} images in json\n'.format(num_missing_images))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PyTorch Training')
  parser.add_argument('-i', '--input_dir', type=str, help='<Required> Input directory where dicom data files are saved', required=True)
  parser.add_argument('-c', '--case_tsv', type=str, help='<Required> Input tsv file with all cases info', required=True)
  parser.add_argument('-s', '--series_tsv', type=str, help='<Required> Input tsv file with all image series info', required=True)
  parser.add_argument('-o', '--output_file', type=str, help='<Required> Output log file', required=True)
  args = parser.parse_args()
  read_open_A1_20221010(args)
