'''
	Main program that summarizes all the CXR data repos

	RKS, 05/13/2022
'''
import os
import argparse
import pydicom
import pandas as pd
# #
num_patients_COVID_19_NY_SBU = 1384
num_patients_COVID_19_AR = 105
num_patients_RICORD_1c = 361

# clinical data paths
RICORD_1c_table_path = "/gpfs_projects/common_data/MIDRC/Release_1c/MIDRC-RICORD-1c Clinical Data Jan 13 2021 .xlsx"



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
	print(in_dir)
	print(out_summ_file)
	# # Useful info
	# # Each patient will have multiple time points
	# # In each time point, there could be multiple scans
	# # For each of these scans, we include the scans with ds.SeriesNumber == 1
	# # 
	# # At the root level, there should be 1384 patients
	# patient_dirs = os.listdir(in_dir)
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	if len(patient_dirs) != num_patients_COVID_19_NY_SBU:
		print('ERROR with num. of patients in COVID_19_NY_SBU')
		print('Got {} case, actual should be {}'.format(len(patient_dirs), num_patients_COVID_19_NY_SBU))
		print('Doing nothing. Returning!')
		return
	df = pd.DataFrame(columns=['patient_id', 'images_good', 'num_good', 'images_bad', 'num_bad'])
	# # Iterate over the patients
	# patient_dirs = ['A002279']	# # for debug
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_bad = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print(patient_root_dir)
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir,filename))]
		# print(time_dirs)
		# print(len(time_dirs))
		# # Iterate over different time points
		for each_time in time_dirs:
			# print('==================================')
			# print(each_time)
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir,filename))]
			# print(scans_dirs)
			# # Here exclude the scans that are difference images
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
					if (0x0008, 0x0060) in ds:
						if ds[0x0008, 0x0060].value == 'CR':
							imgs_good += [each_dcm_file]
						else:
							imgs_bad += [each_dcm_file]
					else:
						imgs_bad += [each_dcm_file]
				# break
			# #
			# break
		# print(imgs_good)
		# print(imgs_bad)
		df.loc[ii] = [each_patient] + [imgs_good] + [len(imgs_good)] + [imgs_bad] + [len(imgs_bad)]
		# #
		# break
	# print(df)
	df.to_csv(out_summ_file, sep='\t', index=False)


def read_COVID_19_AR(in_dir, out_summ_file):
	print(in_dir)
	print(out_summ_file)
	# # Useful info
	# # Each patient will have multiple time points
	# # In each time point, there could be multiple scans and non-CXR data too
	# # For each of these scans, we include the scans with ds.SeriesNumber == 1
	# # 
	# # At the root level, there should be 1384 patients
	# patient_dirs = os.listdir(in_dir)
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	if len(patient_dirs) != num_patients_COVID_19_AR:
		print('ERROR with num. of patients in COVID_19_AR')
		print('Got {} case, actual should be {}'.format(len(patient_dirs), num_patients_COVID_19_NY_SBU))
		print('Doing nothing. Returning!')
		return
	df = pd.DataFrame(columns=['patient_id', 'images_good', 'num_good', 'images_bad', 'num_bad'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_bad = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print(patient_root_dir)
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir,filename))]
		# print(time_dirs)
		# print(len(time_dirs))
		# # Iterate over different time points
		for each_time in time_dirs:
			# print('==================================')
			# print(each_time)
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir,filename))]
			# print(scans_dirs)
			# # Here exclude the scans that are difference images
			# # 
			dcm_files = searchthis(time_root_dir, '.dcm')
			for each_dcm_file in dcm_files:
				# print(each_dcm_file)
				ds = pydicom.read_file(each_dcm_file)
				if (0x0008, 0x1030) in ds:
					# print('(0x0008, 0x1030)')
					# print(ds[0x0008, 0x1030].value)
					if 'CXR14' in ds[0x0008, 0x1030].value or \
					'XR CHEST AP ONLY' in ds[0x0008, 0x1030].value or \
					'XR CHEST AP PORTABLE' in ds[0x0008, 0x1030].value or \
					'XR CHEST PA ONLY' in ds[0x0008, 0x1030].value:
						# # this CXR image
						# print([each_dcm_file, ds[0x0008, 0x1030]])
						imgs_good += [each_dcm_file]
				else:
					# # any image other CXR
					# print([ds.SeriesNumber, ds.AcquisitionNumber])
					imgs_bad += [each_dcm_file]
				# break
			# #
			# break
		# print(imgs_good)
		# print(imgs_bad)
		df.loc[ii] = [each_patient] + [imgs_good] + [len(imgs_good)] + [imgs_bad] + [len(imgs_bad)]
		# #
		# break
	# print(df)
	df.to_csv(out_summ_file, sep='\t', index=False)


def read_open_AI(in_dir, out_summ_file):
	'''
		function to read the unzipped open-AI data repo
	'''
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	print(patient_dirs)
	df = pd.DataFrame(columns=['patient_id', 'images_good', 'num_good', 'images_bad', 'num_bad'])
	# # Iterate over the patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_bad = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		print('==================================')
		print(patient_root_dir)
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir,filename))]
		# # Iterate over different time points
		for each_time in time_dirs:
			print('------------------------')
			print(each_time)
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir,filename))]
			# print(scans_dirs)
			# # Here exclude the scans that are difference images
			# # 
			dcm_files = searchthis(time_root_dir, '.dcm')
			for each_dcm_file in dcm_files:
				print(each_dcm_file)
				ds = pydicom.read_file(each_dcm_file)
				if (0x0008, 0x1030) in ds:
					# print('(0x0008, 0x1030)')
					# print(ds[0x0008, 0x1030].value)
					if 'XR CHEST PA/LATERAL' in ds[0x0008, 0x1030].value:
						# # this CXR image
						# print([each_dcm_file, ds[0x0008, 0x1030]])
						imgs_good += [each_dcm_file]
				else:
					# # any image other CXR
					# print([ds.SeriesNumber, ds.AcquisitionNumber])
					imgs_bad += [each_dcm_file]
				# break
			# #
			# break
		# print(imgs_good)
		# print(imgs_bad)
		df.loc[ii] = [each_patient] + [imgs_good] + [len(imgs_good)] + [imgs_bad] + [len(imgs_bad)]
		# #
		# break
	# print(df)
	df.to_csv(out_summ_file, sep='\t', index=False)



def read_RICORD_1c(in_dir, out_summ_file):
	print(in_dir)
	print(out_summ_file)
	# check number of patients (should be 361)
	patient_dirs = [filename for filename in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir,filename))]
	if len(patient_dirs) != num_patients_RICORD_1c:
		print('ERROR with number of patients in RICORD_1c')
		print(f'Got {len(patient_dirs)} case, actual should be {num_patients_RICORD_1c}')
		print('Doing nothing. Returning!')
		return
	# get patient info (need to install xlrd or openpyxl to read excel)
	# patient_df = pd.read_excel(RICORD_1c_table_path)
	# set up dataframe
	df = pd.DataFrame(columns=['patient_id', 'images','images_info', 'patient_info','repo'])
	# iterate through patients
	for ii, each_patient in enumerate(patient_dirs):
		imgs_good = []
		imgs_good_info = []
		imgs_bad = []
		patient_root_dir = os.path.join(in_dir, each_patient)
		patient_submitter_id = ''
		# print(patient_root_dir)
		time_dirs = [filename for filename in os.listdir(patient_root_dir) if os.path.isdir(os.path.join(patient_root_dir, filename))]
		#print(f'{len(time_dirs)} time directories found')
		for each_time in time_dirs:
			time_root_dir = os.path.join(patient_root_dir, each_time)
			scans_dirs = [filename for filename in os.listdir(time_root_dir) if os.path.isdir(os.path.join(time_root_dir, filename))]
			dcm_files = searchthis(time_root_dir,'.dcm')
			for each_dcm_file in dcm_files:
				ds = pydicom.read_file(each_dcm_file)
				if (0x0008, 0x1030) in ds:
					if ds[0x0008,0x1030].value == "XR CHEST 1 VIEW AP" or 'CHEST 1V':
						# CXR image
						# TODO - check for other tags that indicate good images
						imgs_good += [each_dcm_file]
						if (0x0010, 0x0020) in ds:
							patient_specific_id = ds[0x0010, 0x0020].value.replace('MIDRC-RICORD-1C-','')
							# patient_specific_df = patient_df[patient_df['Anon MRN'].str.contains(patient_specific_id)]
							print(patient_specific_id)
							patient_good_info = [{
								'sex':ds[0x0010,0x0040].value,
								'age':ds[0x0010,0x1010].value,
							}]
							imgs_good_info += [{
								'modality':ds[0x0008,0x0060].value
							}]
					else:
						print(ds[0x0008,0x1030].value)
				else:
					imgs_bad += [each_dcm_file]
		df.loc[ii] = [patient_specific_id] + [imgs_good] + [imgs_good_info] + [patient_good_info] + ['RICORD-1c']
		#break
	df.to_json(out_summ_file, indent=4, orient='table', index=False)



def select_fn(sel_repo):
	if sel_repo[0] == 'COVID_19_NY_SBU':
		read_COVID_19_NY_SBU(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'COVID_19_AR':
		read_COVID_19_AR(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'open_AI':
		read_open_AI(sel_repo[1], sel_repo[2])
	elif sel_repo[0] == 'RICORD_1c':
		read_RICORD_1c(sel_repo[1], sel_repo[2])
	else:
		print('ERROR. Uknown REPO. Nothing to do here.')


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
	


