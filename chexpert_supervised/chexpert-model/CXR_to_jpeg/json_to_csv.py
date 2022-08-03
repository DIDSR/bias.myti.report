import pandas as pd
import numpy as np
import argparse
import pydicom
import cv2
import os


def convert_dicom_to_jpeg(input_file, img_save_loc, csv_save_loc):
    '''
    converts json summary files to csv format, if 
    converts dicom image files to jpeg format and saves in a specified folder
    '''
    
    in_df = pd.read_json(input_file, orient='table')
    #in_df = pd.read_json(input_file)
    out_df = pd.DataFrame(columns=['patient id','dicom file', 'Path', 'CR', 'DX', 'Female', 'Male'])

    for index, row in in_df.iterrows():
        # gather patient and image information ============
        patient_sex = row['patient_info'][0]['sex']
        for i in range(len(row['images'])):
            if not os.path.exists(row['images'][i]):
                # for one patient in open_AI instead of having a list of images, it has a list of individual letters
                # that spells out "ERROR - patient not found in the clinical file"
                print(f"image {row['images'][i]} does not exist")
                continue
            if out_df.empty:
                idx = 0
            else:
                idx = out_df.index.max()+1
            out_df.loc[idx, 'patient id'] = row['patient_id']
            out_df.loc[idx, 'dicom file'] = row['images'][i]
            
            img_modality = row['images_info'][i]['modality']
            
            if patient_sex == 'F':
                out_df.loc[idx, 'Female'] = 1
                out_df.loc[idx, 'Male'] = 0
            elif patient_sex == 'M':
                out_df.loc[idx, 'Female'] = 0
                out_df.loc[idx, 'Male'] = 1
            else:
                out_df.loc[idx, 'Female'] = 0
                out_df.loc[idx, 'Male'] = 0

            if img_modality == 'CR':
                out_df.loc[idx, 'CR'] = 1
                out_df.loc[idx, 'DX'] = 0
            elif img_modality == 'DX':
                out_df.loc[idx, 'CR'] = 0
                out_df.loc[idx, 'DX'] = 1
            else:
                out_df.loc[idx, 'CR'] = 0
                out_df.loc[idx, 'DX'] = 0
            # if working with COVIDGR, we already have jpeg files, and no sex or modality information
            if 'COVIDGR' in input_file:
                out_df.loc[idx, 'dicom file'] = None
                output_file_path = row['images'][i]
                # set all modalities to CR
                out_df.loc[idx, 'CR'] = 1
                continue
                            
            out_loc = os.path.join(img_save_loc, f"{row['repo']}_jpegs")
            
            output_file_path = os.path.join(out_loc, f"{row['patient_id']}_{i}.jpeg")
            out_df.loc[idx, 'dicom file'] = row['images'][i]
            
            out_df.loc[idx, 'Path'] = output_file_path
            # convert dicom to jpeg ==========
            if not os.path.exists(output_file_path):
                print('converting dicom to jpeg...')
                # only convert and save image if that image is not already in jpeg format
                dcm_file = pydicom.dcmread(row['images'][i])
                raw_image = dcm_file.pixel_array
                # Normalize pixels to be in [0, 255].
                rescaled_image = cv2.convertScaleAbs(dcm_file.pixel_array,
                                                    alpha=(255.0/dcm_file.pixel_array.max()))
                # Correct image inversion.
                if dcm_file.PhotometricInterpretation == "MONOCHROME1":
                    rescaled_image = cv2.bitwise_not(rescaled_image)
                # Perform histogram equalization.
                adjusted_image = cv2.equalizeHist(rescaled_image)
                # Save image
                cv2.imwrite(output_file_path, adjusted_image)
            
    # save csv   
    if csv_save_loc:
        # if saving in a different file
        csv_name = input_file.split('/')[-1].replace('.json', '.csv')
        csv_path = os.path.join(csv_save_loc, csv_name)
    else:
        # if saving in the same file as input
        csv_path = input_file.replace('.json','.csv')
    out_df.to_csv(csv_path)

    print(out_df.head(10))
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='summary file', required=True)
    parser.add_argument('-j', '--jpeg_loc', required=True)
    parser.add_argument('-c', '--csv_loc', default=None)
    args = parser.parse_args()

    
    convert_dicom_to_jpeg(args.input, args.jpeg_loc, args.csv_loc)

