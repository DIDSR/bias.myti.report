#!/bin/bash
# # run image cropping on dicom files and save as jpeg files
source /scratch/ravi.samala/anaconda3/envs/venv_python38_2022116/bin/activate
SAVE_DIR=/scratch/yuhang.zhang/OUT/temp/20221010_open_A1_jpegs_crop_test
INPUT_FILE=/scratch/alexis.burgon/2022_CXR/data_summarization/20221010/summary_table__open_A1.json
python ../src/image_cropping.py -s 0 \
                                -d ${SAVE_DIR} \
                                -i ${INPUT_FILE} \
                                -c 0.1 0.1 0.1 0.1