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
'''
import os
import argparse
import pydicom
import pandas as pd
# #
# # Some constants regarding the data repos
num_patients_COVID_19_NY_SBU = 1384
num_patients_COVID_19_AR = 105
num_patients_RICORD_1c = 361
open_AI_MIDRC_table_path = '../data/open_AI_all_20220624.tsv'
open_RI_MIDRC_table_path = '../data/open_RI_all_20220609.tsv'
COVID_19_NY_SBU_TCIA_table_path = '../data/deidentified_overlap_tcia.csv.cleaned.csv_20210806.csv'
COVID_19_AR_TCIA_table_path = '../data/COVID_19_AR_ClinicalCorrelates_July202020.xlsx'
RICORD_1c_annotation_path = "../data/1c_mdai_rsna_project_MwBeK3Nr_annotations_labelgroup_all_2021-01-08-164102_v3.csv"


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
				if (0x0008, 0x1140) in ds:
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
							imgs_good += [each_dcm_file]
							imgs_good_info += [{
								'modality': ds[0x0008, 0x0060].value if (0x0008, 0x0060) in ds else "MISSING",
								'body part examined':ds[0x0018,0x0015].value if (0x0018,0x0015) in ds else "MISSING",
								'view position':ds[0x0018,0x5101].value if (0x0018,0x5101) in ds else "MISSING",
								'pixel spacing':[ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else "MISSING",
								'study date':ds[0x0008,0x0020].value if (0x0008,0x0020) in ds else "MISSING",
								'manufacturer':ds[0x0008,0x0070].value if (0x0008,0x0070) in ds else "MISSING",
								'manufacturer model name':ds[0x0008,0x1090].value if (0x0008,0x1090) in ds else "MISSING",
								'image size': ds.pixel_array.shape
								}]
							patient_good_info = [{
								'sex':"M" if patient_specific_df.iloc[0]['gender_concept_name'] == "Male" else "F",
								'race':"MISSING",
								'ethnicity':"MISSING",
								'COVID_positive':"Yes" if patient_specific_df.iloc[0]['covid19_statuses'] == "positive" else "No",
								'age':patient_specific_df.iloc[0]['age.splits'],
								}]
						else:
							imgs_bad += [each_dcm_file]
					else:
						imgs_bad += [each_dcm_file]
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
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images', 'repo'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
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
				if (0x0008, 0x1030) in ds:
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
						imgs_good_info += [{
							'modality': ds[0x0008, 0x0060].value if (0x0008, 0x0060) in ds else "MISSING",
							'body part examined':ds[0x0018,0x0015].value if (0x0018,0x0015) in ds else "MISSING",
							'view position':ds[0x0018,0x5101].value if (0x0018,0x5101) in ds else "MISSING",
							'pixel spacing':[ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else "MISSING",
							'study date':ds[0x0008,0x0020].value if (0x0008,0x0020) in ds else "MISSING",
							'manufacturer':ds[0x0008,0x0070].value if (0x0008,0x0070) in ds else "MISSING",
							'manufacturer model name':ds[0x0008,0x1090].value if (0x0008,0x1090) in ds else "MISSING",
							'image size': ds.pixel_array.shape
							}]
						patient_good_info = [{
							'sex':patient_specific_df.iloc[0]['SEX'],
							'race':patient_specific_df.iloc[0]['RACE'],
							'ethnicity':"MISSING",
							'COVID_positive':"Yes" if patient_specific_df.iloc[0]['COVID TEST POSITIVE'] == "Y" else "No",
							'age':patient_specific_df.iloc[0]['AGE'],
							}]
				else:
					# # any image other CXR
					imgs_bad += [each_dcm_file]
		df.loc[ii] = [each_patient] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] + ['COVID_19_AR']
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
	# # get patient info
	patient_df = pd.read_csv(open_AI_MIDRC_table_path, sep='\t')
	# # iterate over the dirs
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	print('There are {:d} dirs'.format(len(patient_dirs)))
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images', 'repo'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print([ii, patient_root_dir], flush=True)
		patient_submitter_id = ''
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
				# print(ds)
				if (0x0008, 0x1030) in ds:
					print(ds[0x0008, 0x1030].value)
					if 'XR CHEST PA/LATERAL' in ds[0x0008, 0x1030].value or \
						'XR PORT CHEST 1V' in ds[0x0008, 0x1030].value or \
						'XR CHEST PA AND LATERAL' in ds[0x0008, 0x1030].value:
						# # this CXR image
						imgs_good += [each_dcm_file]
						if (0x0010, 0x0020) in ds:
							patient_specific_id = ds[0x0010, 0x0020].value
							patient_specific_df = patient_df[patient_df['submitter_id'].str.contains(patient_specific_id)]
							print(patient_specific_df)
							# # there should be only 1 row for each patient
							if len(patient_specific_df.index) != 1:
								print('ERROR')
								imgs_good = 'ERROR - patient not found in the clinical file'
								break
							else:
								if len(patient_submitter_id) == 0:
									patient_submitter_id = patient_specific_df.iloc[0]['submitter_id']
								else:
									# # check if the patient name is the same
									if patient_submitter_id != patient_specific_df.iloc[0]['submitter_id']:
										print('ERROR')
										imgs_good = 'ERROR with patient name within each patient dir'
							imgs_good_info += [{
								'modality': ds[0x0008, 0x0060].value if (0x0008, 0x0060) in ds else "MISSING",
								'body part examined':ds[0x0018,0x0015].value if (0x0018,0x0015) in ds else "MISSING",
								'view position':ds[0x0018,0x5101].value if (0x0018,0x5101) in ds else "MISSING",
								'pixel spacing':[ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else "MISSING",
								'study date':ds[0x0008,0x0020].value if (0x0008,0x0020) in ds else "MISSING",
								'manufacturer':ds[0x0008,0x0070].value if (0x0008,0x0070) in ds else "MISSING",
								'manufacturer model name':ds[0x0008,0x1090].value if (0x0008,0x1090) in ds else "MISSING",
								'image size': ds.pixel_array.shape
								}]
							patient_good_info = [{
								'sex':"M" if patient_specific_df.iloc[0]['sex'] == "Male" else "F",
								'race':patient_specific_df.iloc[0]['race'],
								'ethnicity':patient_specific_df.iloc[0]['ethnicity'],
								'COVID_positive':patient_specific_df.iloc[0]['covid19_positive'],
								'age':patient_specific_df.iloc[0]['age_at_index'],
								}]
						else:
							# # 
							imgs_good = '0x0010, 0x0020 NOT FOUND'
					else:
						imgs_bad += [each_dcm_file]
				else:
					# # any image other CXR
					imgs_bad += [each_dcm_file]
		df.loc[ii] = [patient_submitter_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] + ['open_AI']
		# # # for debug
		# if ii == 1:
		# 	break
	# #
	# # save info
	df.to_json(out_summ_file, indent=4, orient='table', index=False)


def read_open_RI(in_dir, out_summ_file):
	'''
		function to read the unzipped open-AI data repo
		use the MIDRC table to get the patient info
	'''
	# # get patient info
	patient_df = pd.read_csv(open_RI_MIDRC_table_path, sep='\t')
	# # iterate over the dirs
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	print('There are {:d} dirs'.format(len(patient_dirs)))
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images', 'repo'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print([ii, patient_root_dir], flush=True)
		patient_submitter_id = ''
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
				# print(ds)
				if (0x0008, 0x1030) in ds:
					if 'CT' not in ds[0x0008, 0x1030].value:
						print(ds[0x0008, 0x1030].value)
					if 'XR CHEST 1 VIEW AP' in ds[0x0008, 0x1030].value or \
						'XR CHEST 2 VIEWS PA AND LATERAL' in ds[0x0008, 0x1030].value or \
						'XR CHEST 1 VIEW AP' in ds[0x0008, 0x1030].value:
						# # this CXR image
						imgs_good += [each_dcm_file]
						if (0x0010, 0x0020) in ds:
							patient_specific_id = ds[0x0010, 0x0020].value
							# print(patient_specific_id)
							patient_specific_df = patient_df[patient_df['submitter_id'].str.contains(patient_specific_id)]
							# print(patient_specific_df)
							# # there should be only 1 row for each patient
							if len(patient_specific_df.index) != 1:
								print('ERROR')
								imgs_good = 'ERROR - patient not found in the clinical file'
								break
							else:
								if len(patient_submitter_id) == 0:
									patient_submitter_id = patient_specific_df.iloc[0]['submitter_id']
								else:
									# # check if the patient name is the same
									if patient_submitter_id != patient_specific_df.iloc[0]['submitter_id']:
										print('ERROR')
										imgs_good = 'ERROR with patient name within each patient dir'
							imgs_good_info += [{
								'modality': ds[0x0008, 0x0060].value if (0x0008, 0x0060) in ds else "MISSING",
								'body part examined':ds[0x0018,0x0015].value if (0x0018,0x0015) in ds else "MISSING",
								'view position':ds[0x0018,0x5101].value if (0x0018,0x5101) in ds else "MISSING",
								'pixel spacing':[ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else "MISSING",
								'study date':ds[0x0008,0x0020].value if (0x0008,0x0020) in ds else "MISSING",
								'manufacturer':ds[0x0008,0x0070].value if (0x0008,0x0070) in ds else "MISSING",
								'manufacturer model name':ds[0x0008,0x1090].value if (0x0008,0x1090) in ds else "MISSING",
								'image size': ds.pixel_array.shape
								}]
							patient_good_info = [{
								'sex':"M" if patient_specific_df.iloc[0]['sex'] == "Male" else "F",
								'race':patient_specific_df.iloc[0]['race'],
								'ethnicity':patient_specific_df.iloc[0]['ethnicity'],
								'COVID_positive':patient_specific_df.iloc[0]['covid19_positive'],
								'age':patient_specific_df.iloc[0]['age_at_index'],
								}]
						else:
							# # 
							imgs_good = '0x0010, 0x0020 NOT FOUND'
					else:
						imgs_bad += [each_dcm_file]
				else:
					# # any image other CXR
					imgs_bad += [each_dcm_file]
		df.loc[ii] = [patient_submitter_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] + ['open_AI']
		# # # for debug
		if ii == 20:
			break
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
		
		patient_good_info = [{
			'sex':ds[0x0010,0x0040].value,
			'age':float(ds[0x0010,0x1010].value.replace('Y','')) if (0x0010,0x1010) in ds else "MISSING",
			'race':'Missing',
			'ethnicity':'Missing',
			'COVID_positive':'Yes'
		}]
		imgs_good_info += [{
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
		}]
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
	# set up dataframe
	df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images', 'repo'])
	# iterate through patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print([ii, patient_root_dir], flush=True)
		patient_submitter_id = ''
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir, filename))]
		for each_time in time_dirs:
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir, filename))]
			dcm_files = searchthis(time_root_dir,'.dcm')
			for each_dcm_file in dcm_files:
				ds = pydicom.read_file(each_dcm_file)
				# print(ds)
				# break
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
					# print(ds)
					# print(each_dcm_file)
					# break
		df.loc[ii] = [patient_specific_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + [len(imgs_good)] + ['RICORD-1c']
		# # # # for debug
		# if ii == 10:
		# 	break
	df.to_json(out_summ_file, indent=4, orient='table', index=False)


def select_fn(sel_repo):
	if sel_repo[0] == 'COVID_19_NY_SBU':
		read_COVID_19_NY_SBU(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'COVID_19_AR':
		read_COVID_19_AR(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'open_AI':
		read_open_AI(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'MIDRC_RICORD_1C':
		read_RICORD_1c(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'open_RI':
		read_open_RI(sel_repo[1], sel_repo[2])
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
		break
