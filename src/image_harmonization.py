import pandas as pd
import numpy as np
import argparse
import pydicom
import cv2
import os
import sys
from multiprocessing import Pool
import tqdm
from skimage.measure import label
from scipy import ndimage

def get_largest_region(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largest_region = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largest_region

def get_dcms(file_path):
    dcms = []
    for p, d, f in os.walk(file_path):
        for file in f:
            if file.endswith('.dcm'):
                fp = os.path.join(p, file)
                dcms.append(fp)
    return dcms

def crop_convert_image_loop(img_info):
    img = img_info[0]
    jpeg_path = img_info[1]
    if os.path.exists(jpeg_path):
        # skip already generated jpegs
        return
    # convert dicom to jpeg
    dcm = pydicom.dcmread(img)
    raw_image = dcm.pixel_array
    
    v_min = np.min(raw_image)
    v_max = np.max(raw_image)
    threds = v_min + 0.9 * (v_max - v_min)
    thred_image = np.where(raw_image > threds, 1, 0)
    thred_largest = get_largest_region(thred_image)
    thred_hole_fill = ndimage.binary_fill_holes(thred_largest)
    
    # Normalize pixels to be in [0,255]
    rescaled_img = cv2.convertScaleAbs(raw_image, alpha=(255.0/raw_image.max()))
    # correct image inversion
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        rescaled_img = cv2.bitwise_not(rescaled_img)
    # perform histogram equalization
    adjusted_image = cv2.equalizeHist(rescaled_img)
    # WIP diaphragm removal
    masked_image = np.multiply(adjusted_image,thred_image)
    # bilateral filtering
    # parameters: diameter = 5, s color = s space = 75
    filtered_image =cv2.bilateralFilter(adjusted_image, 5, 75, 75)
    # save image
    cv2.imwrite(jpeg_path, filtered_image)

def crop_convert_dicom_to_jpeg(args):
    '''
    convert and crop dicom files and save as jpeg files, where the name of the jpeg is patient_id_#
    create conversion_table.json, which holds all of the matched up file names,
    save in img_save_loc
    '''
    print("\nStart image cropping and convert to jpeg")
    input_file = args.input_file
    img_save_loc = args.save_dir
    # # create the save dirctory if not exist
    if not os.path.exists(img_save_loc):
        os.makedirs(img_save_loc)
    print(f"input: {input_file}")
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        return
    # img_info = {} # index: [dicom_file_name, jpeg_file_name]
    img_info = [] # [[dcm, jpeg], [dcm, jpeg], ..]
    in_df = pd.read_json(input_file, orient='table')
    for ii, patient in in_df.iterrows():
            patient_id = patient['patient_id']
            for ii, img in enumerate(patient['images']):
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
    for _ in tqdm.tqdm(pool.imap_unordered(crop_convert_image_loop, img_info[:args.stop_at]), total=args.stop_at):
        pass
    
    # save the conversion information
    conversion_df = pd.DataFrame(img_info, columns=['dicom','jpeg'])
    conversion_table_file = os.path.join(img_save_loc, 'conversion_table.json')
    conversion_df.to_json(conversion_table_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--save_dir',type=str)
    parser.add_argument('-s', '--stop_at',type=int, default=0)
    parser.add_argument('-i', '--input_file', required=True,
    help="json file that contains all the dicom file information")
    parser.add_argument('-c', '--crop_ratios',nargs=4,type=float,default=[0,0,0,0],
    help="Specify 4 cropping ratios on top, bottom, left and right side of the image, no cropping if not specified")
    args = parser.parse_args()    
    crop_convert_dicom_to_jpeg(args)
    print("\nDONE\n")
