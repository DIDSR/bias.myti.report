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
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
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

def convert_image_loop(img_info):
    img = img_info[0]
    jpeg_path = img_info[1]
    #if os.path.exists(jpeg_path):
        # skip already generated jpegs
    #    return
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
    # get # of rows and columns
    rows = dcm.Rows
    columns = dcm.Columns
    # calculate average intensity
    average = np.sum(adjusted_image) / (rows * columns)
    
    # force to only segment on bottom part of the image
    half_img = adjusted_image[round(rows*0.5):,:]
    mask_img = np.zeros((rows,columns)).astype(bool)
    # thresholding
    v_min = np.min(half_img)
    v_max = np.max(half_img)
    threds = v_min + args.threshold * (v_max - v_min)
    thred_image = np.where(half_img > threds, 1, 0)
    # get largest region from segments
    thred_largest = get_largest_region(thred_image)
    # fill holes
    thred_hole_fill = ndimage.binary_fill_holes(thred_largest)
    # morphological filter to smooth edges
    footprint = disk(20)
    thred_hole_fill = closing(thred_hole_fill, footprint)
    thred_hole_fill = ndimage.binary_fill_holes(thred_hole_fill)
    mask_img[round(rows*0.5):,:] = thred_hole_fill
    
    # calculate the ratio of removed parts
    ratio = np.sum(mask_img.astype(int)) / (rows * columns)

    # diaphragm removal
    masked_image = np.multiply(adjusted_image, ~mask_img)
    average_fill = np.multiply(np.full((rows,columns), average).astype(np.uint8), mask_img)
    final_image = np.add(masked_image, average_fill)
    # bilateral filtering
    # parameters: diameter = 5, s color = s space = 75
    filtered_image =cv2.bilateralFilter(final_image, 5, 75, 75)
    # save image
    cv2.imwrite(jpeg_path, filtered_image)
    return average, ratio

def convert_dicom_to_jpeg(args):
    """ Remove bright diaphragm from the image, and convert dicom files to jpeg files.
    """
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
    
    ratios = []
    average_intensity = []
    
    # # parallelize
    pool = Pool(os.cpu_count()-1)
    for average, ratio in tqdm.tqdm(pool.imap_unordered(convert_image_loop, img_info[:args.stop_at]), total=args.stop_at):
        average_intensity.append(average)
        ratios.append(ratio)
        
    # save the conversion information
    average_intensity_df = pd.DataFrame(average_intensity, columns=['intensity'])
    average_intensity_file = os.path.join(img_save_loc, 'average_intensity.csv')
    average_intensity_df.to_csv(average_intensity_file)
    ratio_df = pd.DataFrame(ratios, columns=['ratio'])
    ratio_file = os.path.join(img_save_loc, 'ratios.csv')
    ratio_df.to_csv(ratio_file)
    conversion_df = pd.DataFrame(img_info, columns=['dicom','jpeg'])
    conversion_table_file = os.path.join(img_save_loc, 'conversion_table.json')
    conversion_df.to_json(conversion_table_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--save_dir',type=str)
    parser.add_argument('-s', '--stop_at',type=int, default=0)
    parser.add_argument('-t', '--threshold',type=float, default=1.0)
    parser.add_argument('-i', '--input_file', required=True,
    help="json file that contains all the dicom file information")
    args = parser.parse_args()    
    convert_dicom_to_jpeg(args)
    print("\nDONE\n")
