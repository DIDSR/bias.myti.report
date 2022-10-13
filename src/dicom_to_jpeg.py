import pandas as pd
import numpy as np
import argparse
import pydicom
import cv2
import os
import sys
from multiprocessing import Pool
import tqdm


def get_repo(args):
    if args.betsy:
        # summary_json = f"/scratch/alexis.burgon/2022_CXR/data_summarization/20220823/summary_table__{args.repo}.json"
        # img_save_loc = f"/scratch/alexis.burgon/2022_CXR/CXR_jpegs/{args.repo}"
        summary_json = f"/scratch/ravi.samala/2022_CXR/data_summarization/20221010/20221010_summary_table__{args.repo}.json"
        img_save_loc = f"/scratch/ravi.samala/2022_CXR/data_summarization/20221010/20221010_{args.repo}_jpegs"
    else:
        # summary_json = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/summary_table__{args.repo}.json"
        # img_save_loc = f"/gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823/{args.repo}_jpegs"
        summary_json = f"/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_summary_table__{args.repo}.json"
        img_save_loc = f"/gpfs_projects/ravi.samala/OUT/2022_CXR/data_summarization/20221010/20221010_{args.repo}_jpegs"
    return summary_json, img_save_loc

def get_dcms(file_path):
    dcms = []
    for p, d, f in os.walk(file_path):
        for file in f:
            if file.endswith('.dcm'):
                fp = os.path.join(p, file)
                dcms.append(fp)
    return dcms

def convert_dicom_to_jpeg(args):
    '''
    converts dicom to jpeg files, where the name of the jpeg is patient_id_#
    creates conversion_table.json, which holds all of the matched up file names,
    saves in img_save_loc
    '''
    input_file, img_save_loc = get_repo(args)
    if not os.path.exists(img_save_loc):
        os.makedirs(img_save_loc)
    print(f"for repository {args.repo} using input {input_file}")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return
    img_info = {} # index: [dicom_file_name, jpeg_file_name]
    in_df = pd.read_json(input_file, orient='table')
    for ii, patient in in_df.iterrows():
        patient_id = patient['patient_id']
        for ii, img in enumerate(patient['images']):
            img_info[len(img_info)] = [img, os.path.join(img_save_loc, f"{patient_id}_{ii}.jpg")]
       
    # convert all dicoms (unless stop_at)
    print(f"\nfound {len(img_info)} image files")
    if args.stop_at == 0 or args.stop_at >= len(img_info):
        print('converting all images')
        args.stop_at = len(img_info)
    else:
        print(f"only converting first {args.stop_at}")
    for i in img_info:
        if args.stop_at and i >= args.stop_at:
            break
        sys.stdout.write(f"\r{i+1}/{args.stop_at} files converted")
        #print(f"{i+1}/{args.stop_at}")
        img = img_info[i][0]
        jpeg_path = img_info[i][1]
        if os.path.exists(jpeg_path):
            # skip already generated jpegs
            continue
        # convert dicom to jpeg
        dcm = pydicom.dcmread(img)
        raw_image = dcm.pixel_array
        # Normalize pixels to be in [0,255]
        rescaled_img = cv2.convertScaleAbs(raw_image, alpha=(255.0/raw_image.max()))
        # correct image inversion
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            rescaled_img = cv2.bitwise_not(rescaled_img)
        # perform histogram equalization
        adjusted_image = cv2.equalizeHist(rescaled_img)
        # save image
        cv2.imwrite(jpeg_path, adjusted_image)
    
    # save the conversion information
    conversion_df = pd.DataFrame.from_dict(img_info, orient='index', columns=['dicom','jpeg'])
    conversion_table_file = os.path.join(img_save_loc, 'conversion_table.json')
    conversion_df.to_json(conversion_table_file)

def convert_image_loop(img_info):
    img = img_info[0]
    jpeg_path = img_info[1]
    if os.path.exists(jpeg_path):
        # skip already generated jpegs
        return
    # convert dicom to jpeg
    dcm = pydicom.dcmread(img)
    raw_image = dcm.pixel_array
    # Normalize pixels to be in [0,255]
    rescaled_img = cv2.convertScaleAbs(raw_image, alpha=(255.0/raw_image.max()))
    # correct image inversion
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        rescaled_img = cv2.bitwise_not(rescaled_img)
    # perform histogram equalization
    adjusted_image = cv2.equalizeHist(rescaled_img)
    # save image
    cv2.imwrite(jpeg_path, adjusted_image)

def convert_dicom_to_jpeg_parallel(args):
    '''
    converts dicom to jpeg files, where the name of the jpeg is patient_id_#
    creates conversion_table.json, which holds all of the matched up file names,
    saves in img_save_loc
    '''
    input_file, img_save_loc = get_repo(args)
    if not os.path.exists(img_save_loc):
        os.makedirs(img_save_loc)
    print(f"repository: {args.repo}, \ninput: {input_file}")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return
    # img_info = {} # index: [dicom_file_name, jpeg_file_name]
    img_info = [] # [[dcm, jpeg], [dcm, jpeg], ..]
    in_df = pd.read_json(input_file, orient='table')
    for ii, patient in in_df.iterrows():
        patient_id = patient['patient_id']
        for ii, img in enumerate(patient['images']):
            # img_info[len(img_info)] = [img, os.path.join(img_save_loc, f"{patient_id}_{ii}.jpg")]
            img_info += [[img, os.path.join(img_save_loc, f"{patient_id}_{ii}.jpg")]]
       
    # convert all dicoms (unless stop_at)
    print(f"\nfound {len(img_info)} image files")
    if args.stop_at == 0 or args.stop_at >= len(img_info):
        print('converting all images')
        args.stop_at = len(img_info)
    else:
        print(f"only converting first {args.stop_at}")
    
    # # parallelize
    pool = Pool(os.cpu_count()-1)
    for _ in tqdm.tqdm(pool.imap_unordered(convert_image_loop, img_info), total=len(img_info)):
        pass
    # pool.map(convert_image_loop, img_info)
    
    # save the conversion information
    # conversion_df = pd.DataFrame.from_dict(img_info, orient='index', columns=['dicom','jpeg'])
    conversion_df = pd.DataFrame(img_info, columns=['dicom','jpeg'])
    conversion_table_file = os.path.join(img_save_loc, 'conversion_table.json')
    conversion_df.to_json(conversion_table_file)

if __name__ == '__main__':
    print("\nStarting dicom to jpeg conversion")
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repo', required=True,
                        choices=['open_A1','open_RI','MIDRC_RICORD_1C',
                        'COVID_19_AR','COVID_19_NY_SBU','COVIDGR_10'])
    parser.add_argument('-s', '--stop_at',type=int, default=0)
    parser.add_argument('-b','--betsy', default=False)
    # convert_dicom_to_jpeg(parser.parse_args())
    convert_dicom_to_jpeg_parallel(parser.parse_args())
    print("\nDONE\n")
