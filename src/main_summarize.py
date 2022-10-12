'''
	Main program that summarizes all the CXR data repos

	RKS, 05/13/2022

	# of cases in open-AI data repo = 7,534

	# # # DICOM tags of interest
	# # Sex
	# # Age
	# # Body part examined
	# # Pixel spacing
	# # View position
	# #
	# #
	# # First check by RKS
	# # - MIDRC_RICORD_1C		361 patients
	# # - COVID_19_AR			105 patients
	# #
	# # -----------------------------------------------
	# # Data cleanup notes
	# # -----------------------------------------------
	# # MIDRC_RICORD_1C: initially collected 1257 images
	# # 	Removing all views with view position (ds[0x0018,0x5101]) == 'LL': removed 64 images
	# # 	Removing all views with view position (ds[0x0018,0x5101]) == 'PA': removed 102 images
	# #		Removing above LL and PA resulted in 1091 images
	# #		Manually removed 186 images resulted in 261 patients with 905 images
	# # COVID_19_AR: initially collected 262 images
	# # 	Removing all images with max pixel values > 10000: removed 59 images
	# # 	Removing all images with max pixel values < 2000: removed 26 images
	# #		Manually removed 8 images
	# # open_RI
	# #		Manually removed 58 images
	# # open_AI
	# #		Manually removed 509 images
	# # COVID-19-NY-SBU
	# # 	Removing all views with view position (ds[0x0018,0x5105]) == 'LATERAL' : removed 9 images
	# # 	Manually removed 7 images
'''
import os
import glob
import fnmatch
import argparse
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import datetime
# #
# # Some constants regarding the data repos
num_patients_COVID_19_NY_SBU = 1384
num_patients_COVID_19_AR = 105
num_patients_RICORD_1c = 361
num_images_COVIDGR_10 = 852
open_AI_MIDRC_table_path = '../data/open_AI_all_20220624.tsv'
open_RI_MIDRC_table_path = '../data/open_RI_all_20220609.tsv'
COVID_19_NY_SBU_TCIA_table_path = '../data/deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv'
COVID_19_AR_TCIA_table_path = '../data/COVID_19_AR_ClinicalCorrelates_July202020.xlsx'
RICORD_1c_annotation_path = "../data/1c_mdai_rsna_project_MwBeK3Nr_annotations_labelgroup_all_2021-01-08-164102_v3.csv"
COVIDGR_10_label_path = "../data/COVIDGR_10_severity.csv"
open_A1_Cases = "../data/20221010_open_A1_all_Cases.tsv"
open_A1_Imaging_Studies = "../data/20221010_open_A1_all_Imaging_Studies.tsv"
open_A1_Imaging_Series = "../data/20221010_open_A1_all_Imaging_Series.tsv"
# files to remove:
COVID_19_AR_bad_files_path = "../data/COVID_19_AR__manually_deleted_images.txt"
RICORD_1c_bad_files_path = "../data/MIDRC_RICORD_1c__manually_deleted_images.txt"
open_RI_bad_files_path = "../data/open_RI_manually_deleted_images.txt"
open_AI_bad_files_path = "../data/open_AI_manually_deleted_images.txt"
COVID_19_NY_SBU_bad_files_path = "../data/COVID_19_NY_SBU_manually_deleted_images.txt"
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
				# print(fullpath)
				lis_paths += [fullpath]
	return lis_paths


def read_COVID_19_NY_SBU(in_dir, out_summ_file):
	# # Useful info
	# # Each patient will have multiple time points
	# # In each time point, there could be multiple scans
	# # For each of these scans, we include the scans with ds.SeriesNumber == 1
	# # 
	# # At the root level, there should be 1384 patients
	# patient_dirs = os.listdir(in_dir)
	# #
	# # get patient info
	patient_df = pd.read_csv(COVID_19_NY_SBU_TCIA_table_path, sep=',')
	# # iterate over the dirs
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	print('There are {:d} dirs'.format(len(patient_dirs)))
	if len(patient_dirs) != num_patients_COVID_19_NY_SBU:
		print('ERROR with num. of patients in COVID_19_NY_SBU')
		print('Got {} case, actual should be {}'.format(len(patient_dirs), num_patients_COVID_19_NY_SBU))
		print('Doing nothing. Returning!')
		return
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images', 'repo'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		imgs_bad_info = []
		with open(COVID_19_NY_SBU_bad_files_path, 'r') as fp:
			bad_files = fp.read().split("\n")
		patient_root_dir = os.path.join(in_dir, each_patient)
		print(patient_root_dir)
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir,filename))]
		# # Iterate over different time points
		for each_time in time_dirs:
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir,filename))]
			# # 
			dcm_files = searchthis(time_root_dir, '.dcm')
			for each_dcm_file in dcm_files:
				# print(each_dcm_file)
				ds = pydicom.read_file(each_dcm_file)
				if each_dcm_file in bad_files:
					imgs_bad += [each_dcm_file]
				elif (0x0008, 0x1140) in ds:
					# # this image could be difference image
					# print([ds.SeriesNumber, ds.AcquisitionNumber, ds.ReferencedImageSequence])
					imgs_bad += [each_dcm_file]
				else:
					# # this image is what we might use
					# print([ds.SeriesNumber, ds.AcquisitionNumber])
					if (0x0008, 0x1030) in ds:
						if 'CHEST AP PORT' in ds[0x0008, 0x1030].value or \
							'CHEST AP VIEWONLY' in ds[0x0008, 0x1030].value or \
							'CHEST AP PORTABLE' in ds[0x0008, 0x1030].value or \
							'CHEST AP VIEW ONLY' in ds[0x0008, 0x1030].value or \
							'CHEST ROUTINE PA AP AND LATERAL' in ds[0x0008, 0x1030].value or \
							'CHEST AP CENTRAL LINE PL PORTABLE' in ds[0x0008, 0x1030].value or \
							'CHEST AP INFANT PORTABLE' in ds[0x0008, 0x1030].value or \
							'CHEST ROUTINE PA\/AP AND LATERAL' in ds[0x0008, 0x1030].value:
							patient_specific_id = ds[0x0010, 0x0020].value
							patient_specific_df = patient_df[patient_df['to_patient_id'].str.contains(patient_specific_id)]
							# # there should be only 1 row for each patient
							if len(patient_specific_df.index) != 1:
								print('ERROR')
								imgs_good = 'ERROR - patient not found in the clinical file'
								break
							else:
								# # check if the patient name is the same
								if each_patient != patient_specific_df.iloc[0]['to_patient_id']:
									print('ERROR')
									imgs_good = 'ERROR with patient name within each patient dir'
							if (0x0018, 0x5101) in ds and ds[0x0018,0x5101].value == 'LATERAL':
								imgs_bad += [each_dcm_file]
							else:
								imgs_good += [each_dcm_file]
								patient_info = {
									'sex':"M" if patient_specific_df.iloc[0]['gender_concept_name'] == "MALE" else "F",
									'race':"MISSING",
									'ethnicity':"MISSING",
									'COVID_positive':"Yes" if patient_specific_df.iloc[0]['covid19_statuses'] == "positive" else "No",
									'age':patient_specific_df.iloc[0]['age.splits'],
									}
								# consistent terminology
								patient_info['race'] = race_lookup(patient_info['race'])
								patient_info['ethnicity'] = ethnicity_lookup(patient_info['ethnicity'])
								patient_good_info += [patient_info]
						else:
							imgs_bad += [each_dcm_file]
					else:
						imgs_bad += [each_dcm_file]
					# image information
					img_info = {
						'modality': ds[0x0008, 0x0060].value if (0x0008, 0x0060) in ds else "MISSING",
						'body part examined':ds[0x0018,0x0015].value if (0x0018,0x0015) in ds else "MISSING",
						'view position':ds[0x0018,0x5101].value if (0x0018,0x5101) in ds else "MISSING",
						'pixel spacing':[ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else "MISSING",
						'study date':ds[0x0008,0x0020].value if (0x0008,0x0020) in ds else "MISSING",
						'manufacturer':ds[0x0008,0x0070].value if (0x0008,0x0070) in ds else "MISSING",
						'manufacturer model name':ds[0x0008,0x1090].value if (0x0008,0x1090) in ds else "MISSING",
						'image size': ds.pixel_array.shape
						}
					img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
					if each_dcm_file in imgs_bad:
						imgs_bad_info += [img_info]
					elif each_dcm_file in imgs_good:
						imgs_good_info += [img_info]
					
		df.loc[ii] = [each_patient] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] + ['COVID_19_NY_SBU']
		# # # # for debug
		# if ii == 20:
		# 	break
	# #
	# # save info
	df.to_json(out_summ_file, indent=4, orient='table', index=False)


def read_COVID_19_AR(in_dir, out_summ_file):
	# # Useful info
	# # Each patient will have multiple time points
	# # In each time point, there could be multiple scans and non-CXR data too
	# # For each of these scans, we include the scans with ds.SeriesNumber == 1
	# # 
	# # At the root level, there should be 1384 patients
	# patient_dirs = os.listdir(in_dir)
	# #
	# # get patient info
	patient_df = pd.read_excel(COVID_19_AR_TCIA_table_path, header=1)
	# #
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	if len(patient_dirs) != num_patients_COVID_19_AR:
		print('ERROR with num. of patients in COVID_19_AR')
		print('Got {} case, actual should be {}'.format(len(patient_dirs), num_patients_COVID_19_NY_SBU))
		print('Doing nothing. Returning!')
		return
	print('There are {:d} dirs'.format(len(patient_dirs)))
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		imgs_bad_info = []
		#print(os.listdir("../data"))
		with open(COVID_19_AR_bad_files_path, 'r') as in_file:
			bad_files = in_file.read().split("\n")
		patient_root_dir = os.path.join(in_dir, each_patient)
		print([ii, patient_root_dir], flush=True)
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir,filename))]
		# # Iterate over different time points
		for each_time in time_dirs:
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir,filename))]
			# # Here exclude the scans that are difference images
			# #
			dcm_files = searchthis(time_root_dir, '.dcm')
			
			for each_dcm_file in dcm_files:
				ds = pydicom.read_file(each_dcm_file)
				
				if each_dcm_file in bad_files:
					imgs_bad += [each_dcm_file]
				elif (0x0008, 0x1030) in ds:
					if 'XR CHEST AP PORTABLE' in ds[0x0008, 0x1030].value or \
					'XR CHEST AP ONLY' in ds[0x0008, 0x1030].value or \
					'XR CHEST PA AND LATERAL' in ds[0x0008, 0x1030].value or \
					'XR ACUTE ABDOMINAL SERIES W PA CHEST PORTABLE' in ds[0x0008, 0x1030].value or \
					'XR CHEST PA ONLY' in ds[0x0008, 0x1030].value:
						patient_specific_id = ds[0x0010, 0x0020].value
						patient_specific_df = patient_df[patient_df['PATIENT_ID'].str.contains(patient_specific_id)]
						# # there should be only 1 row for each patient
						if len(patient_specific_df.index) != 1:
							print('ERROR')
							imgs_good = 'ERROR - patient not found in the clinical file'
							break
						else:
							# # check if the patient name is the same
							if each_patient != patient_specific_df.iloc[0]['PATIENT_ID']:
								print('ERROR')
								imgs_good = 'ERROR with patient name within each patient dir'
						imgs_good += [each_dcm_file]
						patient_info = {
							'sex':patient_specific_df.iloc[0]['SEX'],
							'race':patient_specific_df.iloc[0]['RACE'],
							'ethnicity':"MISSING",
							'COVID_positive':"Yes" if patient_specific_df.iloc[0]['COVID TEST POSITIVE'] == "Y" else "No",
							'age':patient_specific_df.iloc[0]['AGE'],
							}
						# consisten terminology
						patient_info['race'] = race_lookup(patient_info['race'])
						patient_info['ethnicity'] = ethnicity_lookup(patient_info['ethnicity'])
						patient_good_info = [patient_info]
				else:
					# # any image other CXR
					imgs_bad += [each_dcm_file]
				img_info = {
					'modality': ds[0x0008, 0x0060].value if (0x0008, 0x0060) in ds else "MISSING",
					'body part examined':ds[0x0018,0x0015].value if (0x0018,0x0015) in ds else "MISSING",
					'view position':ds[0x0018,0x5101].value if (0x0018,0x5101) in ds else "MISSING",
					'pixel spacing':[ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else "MISSING",
					'study date':ds[0x0008,0x0020].value if (0x0008,0x0020) in ds else "MISSING",
					'manufacturer':ds[0x0008,0x0070].value if (0x0008,0x0070) in ds else "MISSING",
					'manufacturer model name':ds[0x0008,0x1090].value if (0x0008,0x1090) in ds else "MISSING",
					'image size': ds.pixel_array.shape
					}
				# consistent terminology
				img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
				if each_dcm_file in imgs_bad:
					imgs_bad_info += [img_info]
				elif each_dcm_file in imgs_good:
					imgs_good_info += [img_info]
				
		df.loc[ii] = [each_patient] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['COVID_19_AR']
		# # # for debug
		# if ii == 20:
		# 	break
	# #
	# # save info
	df.to_json(out_summ_file, indent=4, orient='table', index=False)


def read_open_AI(in_dir, out_summ_file):
	'''
		function to read the unzipped open-AI data repo
		use the MIDRC table to get the patient info
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
	patient_df = pd.read_csv(open_AI_MIDRC_table_path, sep='\t')
	# # iterate over the dirs
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	print('There are {:d} dirs'.format(len(patient_dirs)))
	# df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images', 'repo'])
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
	with open(open_AI_bad_files_path,'r') as infile:
		bad_file_list = infile.read().split('\n')
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		imgs_bad_info = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print('{} of {}: {}'.format(ii, len(patient_dirs), patient_root_dir), flush=True)
		# get all dcms
		dcms = get_dcms(os.path.join(in_dir, each_patient))
		# get patient id -> check if they have any CXR
		for dcm in dcms:
			ds = pydicom.read_file(dcm)
			if (0x0010, 0x0020) in ds:
				patient_id = ds[0x0010, 0x0020].value
				break
		if not patient_id:
			print('patient id not found')
			return
		elif patient_id not in patient_df.values:
			print(f"patient id {patient_id} not in table")
			return
		# check the table to see how many CXR images the patient has
		patient_specific_df = patient_df[patient_df['submitter_id'] == patient_id]
		num_CXR_files = patient_specific_df['_cr_series_file_count'].item() + patient_specific_df['_dx_series_file_count'].item()
		if num_CXR_files == 0:
			# this patient has no CXR, skipping
			continue
		# otherwise, continue sorting through dicom
		CXR_count = 0
		for dcm in dcms:
			# stop if we've found all the CXR files
			if CXR_count == num_CXR_files:
				break
			if dcm in bad_file_list:
				continue
			ds = pydicom.read_file(dcm)
			# remove any not CR or DX
			if ds[0x0008,0x0060].value != 'CR' and ds[0x0008,0x0060].value != 'DX':
				continue
			else:
				CXR_count += 1
			# sorting out bad CXRs
			if (0x0008,0x1030) in ds:
				if 'XR CHEST PA/LATERAL' in ds[0x0008, 0x1030].value or \
					'XR PORT CHEST 1V' in ds[0x0008, 0x1030].value or \
					'XR CHEST PA AND LATERAL' in ds[0x0008, 0x1030].value:
					if (0x0018,0x5101) in ds:
						if 'LL' in ds[0x0018,0x5101].value or 'RL' in ds[0x0018,0x5101].value or 'ABDOMEN' in ds[0x0018,0x5101]:
							imgs_bad.append(dcm)
							continue
						elif (0x0008,0x0015) in ds and ds[0x0008,0x0015].value == '':
							imgs_bad.append(dcm)
							continue
					imgs_good.append(dcm)
				else:
					imgs_bad.append(dcm)
			else:
				imgs_bad.append(dcm)
			# get img info:
			img_info = {key: ds[img_info_dict[key][0],img_info_dict[key][1]].value if img_info_dict[key] in ds else 'MISSING' for key in img_info_dict}
			img_info['pixel spacing'] = [ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else 'MISSING'
			img_info['image size'] = ds.pixel_array.shape
			# consistent terminology
			img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
			# add to appropriate list
			if dcm in imgs_bad:
				imgs_bad_info.append(img_info)
			elif dcm in imgs_good:
				imgs_good_info.append(img_info)
		# get patient info
		patient_info = {
							'sex':"M" if patient_specific_df.iloc[0]['sex'] == "Male" else "F",
							'race':patient_specific_df.iloc[0]['race'],
							'ethnicity':patient_specific_df.iloc[0]['ethnicity'],
							'COVID_positive':patient_specific_df.iloc[0]['covid19_positive'],
							'age':patient_specific_df.iloc[0]['age_at_index'],
							}
		# make sure that patient race and ethnicity have consistent terminology
		patient_info['race'] = race_lookup(patient_info['race'])
		patient_info['ethnicity'] = ethnicity_lookup(patient_info['ethnicity'])
		patient_good_info = [patient_info]
		# add to df
		df.loc[len(df)] = [patient_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['open_AI']
		# # # for debug
		#if ii == 100:
			#print(df.head(10))
			#break
	# #
	# # save info
	df.to_json(out_summ_file, indent=4, orient='table', index=False)
	
	

def read_open_RI(in_dir, out_summ_file):
	'''
		function to read the unzipped open-AI data repo
		use the MIDRC table to get the patient info
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
	patient_df = pd.read_csv(open_RI_MIDRC_table_path, sep='\t')
	# # iterate over the dirs
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	print('There are {:d} dirs'.format(len(patient_dirs)))
	# check that all patients are in the summary tsv
	patient_dirs = [filename for filename in patient_dirs if filename in patient_df.values]
	for patient_dir in patient_dirs:
		spec_df = patient_df.loc[patient_df['submitter_id'] == patient_dir]
		if (spec_df['_cr_series_file_count'].item() + spec_df['_dx_series_file_count'].item()) == 0:
			patient_dirs.remove(patient_dir)
	print(f"There are {len(patient_dirs)} that have CXR images")
	# get manually removed
	with open(open_RI_bad_files_path,'r') as infile:
		bad_file_list = infile.read().split('\n')
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		imgs_bad_info = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print('{} of {}: {}'.format(ii, len(patient_dirs), patient_root_dir), flush=True)
		
		# get all dcms for patient
		dcms = get_dcms(patient_root_dir)
		# get patient id
		patient_id = each_patient
		# find the specific info in df
		if patient_id not in patient_df.values:
			print(f"patient {patient_id} not found in patient_df")
			return
		patient_specific_df = patient_df.loc[patient_df['submitter_id'] == patient_id]
		# skip any patients that don't have any CXR files
		num_CXR_files = patient_specific_df['_cr_series_file_count'].item() + patient_specific_df['_dx_series_file_count'].item()
		# iterate through files
		CXR_count = 0
		for dcm in dcms:
			# find only CXR images
			#if os.path.getsize(dcm) < 3000000:
            	# CT files are (generally) a lot smaller than CXR
				#continue
			ds = pydicom.read_file(dcm)
			if ds[0x0008, 0x0060].value != 'CR' and ds[0x0008, 0x0060].value != 'DX':
				# not CXR
				continue
			# otherwise, is a CXR image
			CXR_count += 1
			# screen out bad CXR images
			if dcm in bad_file_list:
				imgs_bad.append(dcm)
			elif (0x0018,0x5101) in ds:
				if 'LL' in ds[0x0018,0x5101].value:
					imgs_bad.append(dcm)
			else:
				imgs_good.append(dcm)
			if (0x0008, 0x1030) in ds:
					if 'XR CHEST 1 VIEW AP' in ds[0x0008, 0x1030].value or \
						'XR CHEST 2 VIEWS PA AND LATERAL' in ds[0x0008, 0x1030].value or \
						'XR CHEST 1 VIEW AP' in ds[0x0008, 0x1030].value:
						if (0x0018,0x5101) in ds and 'LL' in ds[0x0018,0x5101].value:
							imgs_bad.append(dcm)
						else:
							imgs_good.append(dcm)
					else:
						imgs_bad.append(dcm)
			else:
				imgs_bad.append(dcm)
			# get file information
			img_info = {key: ds[img_info_dict[key][0],img_info_dict[key][1]].value if img_info_dict[key] in ds else 'MISSING' for key in img_info_dict}
			img_info['pixel spacing'] = [ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else 'MISSING'
			img_info['image size'] = ds.pixel_array.shape
			# consistent terminology
			img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
			# add to appropriate list
			if dcm in imgs_bad:
				imgs_bad_info.append(img_info)
			elif dcm in imgs_good:
				imgs_good_info.append(img_info)
			# break if we've found all of the CXR images
			if CXR_count == num_CXR_files:
				break
		# get patient info
		if CXR_count != num_CXR_files:
			print("didn't find all of the CXR files!")
			return
		patient_good_info = [{
							'sex':"M" if patient_specific_df.iloc[0]['sex'] == "Male" else "F",
							'race':patient_specific_df.iloc[0]['race'],
							'ethnicity':patient_specific_df.iloc[0]['ethnicity'],
							'COVID_positive':patient_specific_df.iloc[0]['covid19_positive'],
							'age':patient_specific_df.iloc[0]['age_at_index'],
							}]
		# make sure that patient race and ethnicity have consistent terminology
		patient_good_info[0]['race'] = race_lookup(patient_good_info[0]['race'])
		patient_good_info[0]['ethnicity'] = ethnicity_lookup(patient_good_info[0]['ethnicity'])

		df.loc[ii] = [patient_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['open_RI']
		# # for debug
		#if ii == 100:
			#break
	# #
	# # save info
	df.to_json(out_summ_file, indent=4, orient='table', index=False)


def extract_RICORD_1c_info(ds, annotation_df):
	patient_good_info = []
	imgs_good_info = []
	# match file to annotation info
	series_date = ds[0x0008,0x0021].value
	series_date = series_date[4:6] + '/' +series_date[6:] + '/' + series_date[0:4]
	if (0x0010, 0x0020) in ds:
		patient_specific_id = ds[0x0010, 0x0020].value.replace('MIDRC-RICORD-1C-','')
		patient_specific_df = annotation_df.loc[annotation_df['Anon MRN'] == patient_specific_id]
		series_specific_df = patient_specific_df.loc[patient_specific_df['Anon TCIA Study Date'] == series_date]
		class_dict = []
		grade_dict = []
		# print(series_specific_df)
		for index,row in series_specific_df.iterrows():
			class_dict += [{
				'reader id':row['Reader_id'],
				'reader name':row['Reader_name'],
				'class id':row['CLASS_labelId'],
				'class label':row['CLASS_label'],
				'class description':row['CLASS_label_description']
			}]
			grade_dict += [{
				'reader id':row['Reader_id'],
				'reader name':row['Reader_name'],
				'grade id':row['GRADE_labelId'],
				'grade label':row['GRADE_label'],
				'grade description':row['GRADE_label_description']
			}]
		
		patient_info = {
			'sex':ds[0x0010,0x0040].value,
			'age':float(ds[0x0010,0x1010].value.replace('Y','')) if (0x0010,0x1010) in ds else "MISSING",
			'race':'Missing',
			'ethnicity':'Missing',
			'COVID_positive':'Yes'
		}
		patient_info['race'] = race_lookup(patient_info['race'])
		patient_info['ethnicity'] = ethnicity_lookup(patient_info['ethnicity'])
		patient_good_info = [patient_info]
		imgs_info = {
			'modality':ds[0x0008,0x0060].value,
			'body part examined':ds[0x0018,0x0015].value,
			'view position':ds[0x0018,0x5101].value,
			'pixel spacing':[ds[0x0028,0x0030].value[0], ds[0x0028,0x0030].value[1]] if (0x0028,0x0030) in ds else "MISSING",
			'study date':ds[0x0008,0x0020].value,
			'manufacturer':ds[0x0008,0x0070].value if (0x0008,0x0070) in ds else "MISSING",
			'manufacturer model name':ds[0x0008,0x1090].value if (0x0008,0x1090) in ds else "MISSING",
			'image size':ds.pixel_array.shape,
			'classification':class_dict,
			'grade':grade_dict
		}
		imgs_info['manufacurter'] = manufacturer_lookup(imgs_info['manufacturer'])
		imgs_good_info += [imgs_info]
	return patient_specific_id, patient_good_info, imgs_good_info


def read_RICORD_1c(in_dir, out_summ_file):
	# check number of patients (should be 361)
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	if len(patient_dirs) != num_patients_RICORD_1c:
		print('ERROR with number of patients in RICORD_1c')
		print(f'Got {len(patient_dirs)} case, actual should be {num_patients_RICORD_1c}')
		print('Doing nothing. Returning!')
		return
	annotation_df = pd.read_csv(RICORD_1c_annotation_path)
	# # read the file with manually excluded images
	with open(RICORD_1c_bad_files_path, 'r') as in_file:
		bad_files = in_file.read().split("\n")
	# set up dataframe
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
	# iterate through patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		imgs_bad_info = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print([ii, patient_root_dir], flush=True)
		patient_submitter_id = ''
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir, filename))]
		for each_time in time_dirs:
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir, filename))]
			dcm_files = searchthis(time_root_dir,'.dcm')
			for each_dcm_file in dcm_files:
				if each_dcm_file in bad_files:
					imgs_bad += [each_dcm_file]
					continue
				ds = pydicom.read_file(each_dcm_file)
				# # Data cleanup
				if ds[0x0018,0x5101].value == "LL" or ds[0x0018,0x5101].value == "PA":
					continue
				if (0x0008, 0x1030) in ds:
					if ds[0x0008,0x1030].value == "XR CHEST 1 VIEW AP" or \
							ds[0x0008,0x1030].value == 'CHEST 1V' or \
							ds[0x0008,0x1030].value == 'XR CHEST 2 VIEWS PA AND LATERAL':
						imgs_good += [each_dcm_file]
						patient_specific_id, patient_good_info, imgs_good_info1 = extract_RICORD_1c_info(ds, annotation_df)
						imgs_good_info += imgs_good_info1
					else:
						print(ds[0x0008,0x1030].value)
				elif ds[0x0008,0x0060].value == 'CR' and ds[0x0008,0x0070].value == 'Agfa' and ds[0x0008,0x1090].value == 'CR 85':
					imgs_good += [each_dcm_file]
					patient_specific_id, patient_good_info, imgs_good_info1 = extract_RICORD_1c_info(ds, annotation_df)
					imgs_good_info += imgs_good_info1
				else:
					imgs_bad += [each_dcm_file]
		df.loc[ii] = [patient_specific_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['RICORD-1c']
		# # # # for debug
		# if ii == 10:
		# 	break
	print('Num. of images = {}'.format(len(df.index)))
	df.to_json(out_summ_file, indent=4, orient='table', index=False)


def read_COVIDGR_10(in_dir, out_summ_file):
	'''
	in_dir: root dir for the COVIDGR_1.0 where "in_dir"/N and "in_dir"/P will
			have normal and positive classes
	Description of repo: Under a close collaboration with an expert radiologist 
	team of the Hospital Universitario San Cecilio, the COVIDGR-1.0 dataset of patients' 
	anonymized X-ray images has been built. 754 images have been collected following a 
	strict labeling protocol. They are categorized into 377 positive cases and 377 negative 
	cases. Positive images correspond to patients who have been tested positive for COVID-19 
	using RT-PCR within a time span of at most 24h between the X-ray image and the test. 
	Every image has been taken using the same type of equipment and with the same 
	format: only the posterior-anterior view is considered.
	'''
	if os.path.exists(os.path.join(in_dir, 'N')) and os.path.exists(os.path.join(in_dir, 'P')):
		# # proceeed
		# # get patient info
		patient_df = pd.read_csv(COVIDGR_10_label_path, sep=',', header=0)
		print(patient_df)
		patient_imgs = []
		for root, dirnames, filenames in os.walk(in_dir):
			for filename in fnmatch.filter(filenames, '*.jpg'):
				patient_imgs.append(os.path.join(root, filename))
		if len(patient_imgs) != num_images_COVIDGR_10:
			print('ERROR with num. of images in COVIDGR_1.0')
			print('Got {} case, actual should be {}'.format(len(patient_imgs), num_images_COVIDGR_10))
			print('Doing nothing. Returning!')
			return
		print('There are {:d} images'.format(len(patient_imgs)))
		df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		imgs_bad_info = []
		# # Iterate over N patients
		patient_imgs = []
		for root, dirnames, filenames in os.walk(os.path.join(in_dir, 'N')):
			for filename in fnmatch.filter(filenames, '*.jpg'):
				patient_imgs.append(os.path.join(root, filename))
		print('Num. of N patients/images = {}'.format(len(patient_imgs)))
		ii = 0
		for each_patient in patient_imgs:
			# # 
			img = np.asarray(Image.open(each_patient))
			imgs_good = [each_patient]
			imgs_info = {
				'modality': "MISSING",
				'body part examined':"MISSING",
				'view position':"PA",
				'pixel spacing':"MISSING",
				'study date':"MISSING",
				'manufacturer':"MISSING",
				'manufacturer model name':"MISSING",
				'image size': img.shape
				}
			imgs_info['manufacturer'] = manufacturer_lookup(imgs_info['manufacturer'])
			imgs_good_info += [imgs_info]
			patient_info = {
				'sex':"MISSING",
				'race':"MISSING",
				'ethnicity':"MISSING",
				'COVID_positive':"No",
				'age':"MISSING",
				}
			patient_info['race'] = race_lookup(patient_info['race'])
			patient_info['ethnicity'] = ethnicity_lookup(patient_info['ethnicity'])
			patient_good_info = [patient_info]
			df.loc[ii] = [os.path.basename(each_patient).split('.')[0]] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['COVIDGR_10']
			ii += 1
		# # Iterate over P patients
		patient_imgs = []
		for root, dirnames, filenames in os.walk(os.path.join(in_dir, 'P')):
			for filename in fnmatch.filter(filenames, '*.jpg'):
				patient_imgs.append(os.path.join(root, filename))
		print('Num. of P patients/images = {}'.format(len(patient_imgs)))
				# # 
		for each_patient in patient_imgs:
			# # 
			specific_patient_df = patient_df.loc[patient_df['Name'] == os.path.basename(each_patient).split('.')[0]].reset_index()
			if len(specific_patient_df.index) == 0:
				print('{} info missing. Skipping image'.format(os.path.basename(each_patient).split('.')[0]))
				continue
			# print(os.path.basename(each_patient).split('.')[0])
			# print(specific_patient_df)
			img = np.asarray(Image.open(each_patient))
			imgs_good = [each_patient]
			imgs_info = {
				'modality': "MISSING",
				'body part examined':"MISSING",
				'view position':"PA",
				'pixel spacing':"MISSING",
				'study date':"MISSING",
				'manufacturer':"MISSING",
				'manufacturer model name':"MISSING",
				'image size': img.shape
				}
			imgs_info['manufacturer'] = manufacturer_lookup(imgs_info['manufacturer'])
			imgs_good_info = [imgs_info]
			patient_info = {
				'sex':"MISSING",
				'race':"MISSING",
				'ethnicity':"MISSING",
				'COVID_positive':"Yes",
				'grade label':specific_patient_df.at[0, 'Severity'],
				'age':"MISSING",
				}
			patient_info['race'] = race_lookup(patient_info['race'])
			patient_info['ethnicity'] = ethnicity_lookup(patient_info['ethnicity'])
			patient_good_info = [patient_info]
			df.loc[ii] = [os.path.basename(each_patient).split('.')[0]] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['COVIDGR_10']
			ii += 1
		# # save info
		df.to_json(out_summ_file, indent=4, orient='table', index=False)
	else:
		print('DOES NOT EXIST ERROR: ' + os.path.join(in_dir, 'N'))
		return
	
	
def read_open_A1_20221010(in_dir, out_summ_file):
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
	patient_df = pd.read_csv(open_A1_Cases, sep='\t')
	# img_study_df = pd.read_csv(open_A1_Imaging_Studies, sep='\t')
	img_series_df = pd.read_csv(open_A1_Imaging_Series, sep='\t')
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
	print('There are {} patients'.format(len(patient_df.index)))
	# # iterate over the patient-id
	num_patients_to_json = 0
	num_images_to_json = 0
	num_series_to_json = 0
	num_missing_images = 0
	for idx, patient_row in patient_df.iterrows():
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
				# print([patient_id, study_row['case_ids_0'], study_row['study_uid']])
				patient_study_path = None
				patient_study_path1 = os.path.join(in_dir, study_row['case_ids_0'], study_row['study_uid_0'], study_row['series_uid'])
				patient_study_path2 = os.path.join(in_dir, study_row['study_uid_0'], study_row['series_uid'])
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
			df.loc[len(df)] = [patient_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] +[imgs_bad] + [imgs_bad_info] + ['open_A1']
			num_patients_to_json += 1
		# # # for debug
		# if num_patients_to_json == 10:
		# 	# print(df.head(10))
		# 	break
		# break
	# #
	# # save info
	print('Saving {} patients to json'.format(num_patients_to_json))
	print('Saving {} images to json'.format(num_images_to_json))
	print('Saving {} series to json'.format(num_series_to_json))
	print('Missing {} images to json'.format(num_missing_images))
	df.to_json(out_summ_file, indent=4, orient='table', index=False)
	pre, ext = os.path.splitext(out_summ_file)
	out_log_file = pre + '.log'
	print('Log file saved at: ' + out_log_file)
	print('json file saved at: ' + out_summ_file)
	with open(out_log_file, 'w') as fp:
		fp.write(str(datetime.datetime.now()) + '\n')
		fp.write('Saved {} patients in json\n'.format(num_patients_to_json))
		fp.write('Saved {} images in json\n'.format(num_images_to_json))
		fp.write('Saved {} series in json\n'.format(num_series_to_json))
		fp.write('Missed {} images in json\n'.format(num_missing_images))
	# df.to_csv(out_summ_file + '.tsv', sep = '\t', index=False)




def select_fn(sel_repo):
	print('Working on {}'.format(sel_repo[0]))
	if sel_repo[0] == 'COVID_19_NY_SBU':
		read_COVID_19_NY_SBU(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'COVID_19_AR':
		read_COVID_19_AR(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'open_A1':
		# read_open_AI(sel_repo[1], sel_repo[2])
		read_open_A1_20221010(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'MIDRC_RICORD_1C':
		read_RICORD_1c(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'open_R1':
		read_open_RI(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'COVIDGR_10':
		read_COVIDGR_10(sel_repo[1], sel_repo[2])
	else:
		print('ERROR. Unknown REPO. Nothing to do here.')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='PyTorch Training')
	parser.add_argument('-i', '--input_dir_list', action='append', help='<Required> List of input dirs', required=True, default=[])
	parser.add_argument('-n', '--names_list', action='append', help='<Required> List of data repos name', required=True, default=[])
	parser.add_argument('-o', '--output_list', action='append', help='<Required> List of output log files', required=True, default=[])
	args = parser.parse_args()
	
	# # iterate over each data repo
	for each_repo in zip(args.names_list, args.input_dir_list, args.output_list):
		select_fn(each_repo)
		# break
