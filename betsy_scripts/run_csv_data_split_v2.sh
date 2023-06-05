#!/bin/bash
#split the dataset for each subgroup (sex, race) to introduce bias in training data
#take a csv file as input and output 2 csv files that contain splitted datasets
#can use this script to split dataset in bias amplification direct approachs (#2a,2b)

#NOTES: currently can split based on sex, race
SUBGROUP="sex" #options include sex, race
INPUT_FILE="train.csv"
OUTOUT_1="train_1.csv"
OUTOUT_2="train_2.csv"
for BATCH in 0
do
for RAND in 0
do
IN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}/RAND_${RAND}
SAVE_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}/RAND_${RAND}
python ../src/csv_data_split_v2.py --test_subgroup ${SUBGROUP} \
                             --in_dir ${IN_DIR} \
                             --save_dir ${SAVE_DIR} \
                             --input_file ${INPUT_FILE} \
                             --output_1 ${OUTOUT_1} \
                             --output_2 ${OUTOUT_2}
done
done