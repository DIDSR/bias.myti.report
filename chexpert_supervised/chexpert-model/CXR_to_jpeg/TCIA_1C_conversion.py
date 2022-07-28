import pandas as pd
import numpy as np
import argparse
import pydicom
import cv2
import os

def convert_dicom_to_jpeg(input_file, img_save_loc, csv_save_loc):
    '''
    converts dicom image files to jpeg format and saves in a specified folder
    '''
    in_df = pd.read_json(input_file, orient='table')
    out_df = pd.DataFrame(columns=['patient id','dicom file', 'jpeg file', 'CR', 'DX', 'Female', 'Male'])

    for index, row in in_df.iterrows():
        # gather patient and image information ============
        patient_sex = row['patient_info'][0]['sex']
        for i in range(len(row['images'])):
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

            if img_modality == 'CR':
                out_df.loc[idx, 'CR'] = 1
                out_df.loc[idx, 'DX'] = 0
            elif img_modality == 'DX':
                out_df.loc[idx, 'CR'] = 0
                out_df.loc[idx, 'DX'] = 1

            # convert dicom to jpeg ==========
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
            output_file_path = os.path.join(img_save_loc, f"{row['patient_id']}_{i}.jpeg")
            out_df.loc[idx, 'jpeg file'] = output_file_path
            cv2.imwrite(output_file_path, adjusted_image)
            
    # save csv
    out_df.to_csv(os.path.join(csv_save_loc, 'TCIA_1C_jpeg_summary.csv'))

    print(out_df.head(10))
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='summary file', required=True)
    parser.add_argument('-j', '--jpeg_loc', required=True)
    parser.add_argument('-c', '--csv_loc', required=True)
    args = parser.parse_args()

    
    convert_dicom_to_jpeg(args.input, args.jpeg_loc, args.csv_loc)

