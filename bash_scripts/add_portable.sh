#!/bin/bash
#for RAND in 0 1 2 3 4 5 6 7 8 9
#do
INPUT_CSV=/gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/subgroup_size_1_image/75w25b/validation_1_image.csv
OUTPUT_CSV=/gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/subgroup_size_1_image/75w25b/validation_1_image_portable.csv
python ../src/portable_or_nonportable.py -i ${INPUT_CSV} \
                                         -o ${OUTPUT_CSV}
#done