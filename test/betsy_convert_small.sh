#!/bin/bash
# # to run on besty02:
# # salloc --partition=cpu_short
# # sh betsy_convert_small

## NOTE: betsy env doesn't seem to have cv2 rn

echo Converting dicom to jpeg
source /home/alexis.burgon/anaconda3/envs/hpc/bin/activate
python dicom_to_jpeg.py -r COVID_19_AR \
                        -s 20 \
                        -b True
echo Done