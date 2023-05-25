#!/bin/bash
IN_DIR="/scratch/yuhang.zhang/OUT/latent_space_run_2a/"
SAVE_DIR="/scratch/yuhang.zhang/OUT/latent_space_run_2a/"
INPUT_FILE="train.csv"
OUTOUT_1="train_1_test.csv"
OUTOUT_2="train_2_test.csv"
python ../src/split_train.py --in_dir ${IN_DIR} \
                             --save_dir ${SAVE_DIR} \
                             --input_file ${INPUT_FILE} \
                             --output_1 ${OUTOUT_1} \
                             --output_2 ${OUTOUT_2}