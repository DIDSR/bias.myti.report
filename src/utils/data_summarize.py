import os
import glob
import fnmatch
import argparse
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import datetime
import time
import json


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

def save_to_file(data:list, filepath:str):
    if len(data) > 0:
        with open(filepath, "a") as file:
            for d in data:
                json.dump(d, file)
                file.write(os.linesep)
    return
	
def read_open_A1_20221010(args):
  '''
  using the imaging data and the associated MIDRC tsv files downloaded on 20221010

  Info on supporting files:
    ../data/20221010_open_A1_all_Cases.tsv: get patient-level info (submitter_id, sex, age, race, COVID_status)
    ../data/20221010_open_A1_all_Imaging_Studies.tsv: for a submitter_id (case_ids_0), use the study_modality_0
      identify the study_uid which is the subdirectory name. However, sometimes, the study_uid is the main directory
  
  11/02/2022: works for both open-A1 and open-R1
  '''
  # information to gather (pixel spacing and img size done separately)
  img_info_dict = {
    'modality':(0x0008,0x0060),
    'body part examined':(0x0018,0x0015),
    'view position':(0x0018,0x5101),
    'study date':(0x0008,0x0020),
    'manufacturer':(0x0008,0x0070),
    'manufacturer model name':(0x0008,0x1090)}
   # get patient info
  patient_df = pd.read_csv(args.case_tsv, sep='\t')
  img_series_df = pd.read_csv(args.series_tsv, sep='\t')
  #df = pd.DataFrame(columns=['patient_id', 'images', 'images_info', 'patient_info', 'num_images','bad_images', 'bad_images_info', 'repo'])
  print('There are {} patients'.format(len(patient_df.index)))
  # remove series that are not in the chosen modalities
  img_series_df = img_series_df[img_series_df['modality'].isin(modality_choices)].copy()
  # remove patients with no valid series
  patient_df = patient_df[patient_df['submitter_id'].isin(img_series_df['case_ids_0'])].copy()
  patient_df.set_index("submitter_id", inplace=True) # set patient id as index
  print(f"Filtered to {len(patient_df)} patients based on series modality")
  total_patients = img_series_df["case_ids_0"].nunique()
  # # iterate over the patient-id
  num_patients_to_json = 0
  num_images_to_json = 0
  num_series_to_json = 0
  num_missing_images = 0
  
  patient_info_list = []
  
  # create the output file
  if args.overwrite:
      with open(args.output_file, "w") as file:
          pass
  else:
      with open(args.output_file, "x") as file:
          pass
  
  for i, (patient_id, df_patient) in enumerate(img_series_df.groupby("case_ids_0")):
    print(f"{i}/{total_patients} ({((i/total_patients)*100):.2f}%)", end="\r")
    #if num_patients_to_json > 0 and num_patients_to_json % 100 == 0:
    #    print('Processed {} patients with {} series and {} images so far...'.format(num_patients_to_json, num_series_to_json, num_images_to_json))
    
    #imgs_good = []
    #imgs_good_info = []
    #imgs_bad = []
    #imgs_bad_info = []
    patient_skip = True
    # #
    #patient_id = patient_row['submitter_id']
    # # identify the dir/sub-dir based on the Imaging_Studies file
    #df_patient = img_series_df.loc[img_series_df['case_ids_0'] == patient_id]
    patient_info = {
        "patient_id": patient_id,
        "images" : [],
        "images_info":[],
        "patient_info":[],
        "num_images":None,
        "bad_images":[],
        "bad_images_info":[],
        "repo":"open-A1",
        }
    for study_idx, study_row in df_patient.iterrows():
        patient_study_path1 = os.path.join(args.input_dir, study_row['case_ids_0'], str(study_row['study_uid_0']), str(study_row['series_uid']))
        patient_study_path2 = os.path.join(args.input_dir, str(study_row['study_uid_0']), str(study_row['series_uid']))
        if os.path.exists(patient_study_path1):
          patient_study_path = patient_study_path1
        elif os.path.exists(patient_study_path2):
          patient_study_path = patient_study_path2
        else:
          num_missing_images += 1
          continue # skip invalid series path
        
        patient_skip = False
        # # get the dicom info here
        # # there should be at least 1 dicom file in this folder
        dcm_files = glob.glob(os.path.join(patient_study_path, '*.dcm'))
        num_images_to_json += len(dcm_files)
        num_series_to_json += 1
        for each_dcm in dcm_files:
          ds = pydicom.read_file(each_dcm)
          patient_info['images'].append(each_dcm)
          # get img info:
          img_info = {key: ds[img_info_dict[key][0],img_info_dict[key][1]].value if img_info_dict[key] in ds else 'MISSING' for key in img_info_dict}
          img_info['pixel spacing'] = [ds[0x0018,0x1164].value[0], ds[0x0018,0x1164].value[1]] if (0x0018,0x1164) in ds else 'MISSING'
          img_info['image size'] = ds.pixel_array.shape
          img_info['manufacturer'] = manufacturer_lookup(img_info['manufacturer'])
          patient_info['images_info'].append(img_info)

    # # create patient-level info
    patient_row = patient_df.loc[patient_id]
    
    patient_info["patient_info"] = {
            'sex':"M" if patient_row['sex'] == "Male" else "F" if patient_row['sex'] == "Female" else "Unknown",
            'race':race_lookup(patient_row['race']),
            'ethnicity':ethnicity_lookup(patient_row['ethnicity']),
            'COVID_positive':patient_row['covid19_positive'],
            'age':patient_row['age_at_index'],
            }
    
    if len(patient_info['images']) == 0: # no images for this patient
        continue
    patient_info['num_images'] = len(patient_info['images'])

    num_patients_to_json += 1
    patient_info_list.append(patient_info)
    # save information to file
    if num_patients_to_json % args.save_every == 0:
        save_to_file(patient_info_list, args.output_file)
        patient_info_list = []
    
    
  print(f"{i+1}/{total_patients} ({(((i+1)/total_patients)*100):.2f}%)", end="\r")
  # # print summary info and save output files
  print('Saving {} patients to json'.format(num_patients_to_json))
  print('Saving {} images to json'.format(num_images_to_json))
  print('Saving {} series to json'.format(num_series_to_json))
  print('Missing {} images'.format(num_missing_images))
  # read in the json file and convert the format
  with open(args.output_file, "r") as file:
      output_information = [json.loads(line) for line in file]
      
  # convert the output information to a dataframe
  df = pd.DataFrame.from_records(output_information)
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
  print("Starting...")
  start_time = time.time()
  parser = argparse.ArgumentParser(description='PyTorch Training')
  parser.add_argument('-i', '--input_dir', type=str, help='<Required> Input directory where dicom data files are saved', required=True)
  parser.add_argument('-c', '--case_tsv', type=str, help='<Required> Input tsv file with all cases info', required=True)
  parser.add_argument('-s', '--series_tsv', type=str, help='<Required> Input tsv file with all image series info', required=True)
  parser.add_argument('-o', '--output_file', type=str, help='<Required> Output log file', required=True)
  parser.add_argument("--overwrite", default=False, action="store_true", help="(Optional) pass to overwrite existing output files")
  parser.add_argument("--save-every", dest="save_every", type=int, default=20, help="(Optional; default=20) How often to save information; helps with memory issues.")
  args = parser.parse_args()
  read_open_A1_20221010(args)
  print(f"complete in {((time.time()-start_time)/60):.2f} min")
