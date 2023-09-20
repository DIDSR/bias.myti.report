#!/bin/bash
# # run image cropping on dicom files and save as jpeg files
SAVE_DIR=/scratch/yuhang.zhang/OUT/temp/20221010_open_A1_jpegs_crop_5
# determine the ratio of the region to be cropped at each side
TOP=0.1
BOTTOM=0.2
LEFT=0.05
RIGHT=0.95
echo Cropping and Converting dicom to jpeg
python ../crop_dicom_to_jpeg.py -r open_A1 \
                        -s 0 \
                        --save_dir ${SAVE_DIR} \
                        --top ${TOP} \
                        --bottom ${BOTTOM} \
                        --left ${LEFT} \
                        --right ${RIGHT} \
                        -b True
echo Done