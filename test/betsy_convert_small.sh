#!/bin/bash
# # to run on besty02:
# # salloc --partition=cpu_short
# # ./betsy_convert_small

echo Converting 20 dicom to jpeg
source /home/alexis.burgon/anaconda3/envs/hpc/bin/activate
python dicom_to_jpeg.py -r open_AI \
                        -s 20 \
                        -b True
echo Done