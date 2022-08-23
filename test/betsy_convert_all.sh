#!/bin/bash
# # to run on besty02:
# # salloc --partition=cpu_short
# # ./betsy_convert_all

echo Converting dicom to jpeg
source /home/alexis.burgon/anaconda3/envs/hpc/bin/activate
python dicom_to_jpeg.py -r COVID_19_AR \
                        -s 0 \
                        -b True
echo Done