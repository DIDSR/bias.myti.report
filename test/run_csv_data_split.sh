#!/bin/bash
#split the dataset evenly for each subgroup (sex, race and COVID)
#take a csv file as input and output 2 csv files that contain splitted datasets
#can use this script to split dataset in bias amplification indirect approachs (#1b)

#NOTES: currently automatically split based on sex, race and COVID
INPUT_FILE="train.csv"
declare -a FRACTION_ARRAY=('0.5' '0.75' '0.9' '1')
FRACTION=0.5 # valid range from 0 to 1
for BATCH in 0
do
for RAND in 0
do
for FRACTION in ${FRACTION_ARRAY[@]}
do
IN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}/RAND_${RAND}
SAVE_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}/RAND_${RAND}
python ../src/csv_data_split.py --fraction ${FRACTION} \
                                --test_subgroup Black White \
                                --in_dir ${IN_DIR} \
                                --save_dir ${SAVE_DIR} \
                                --input_file ${INPUT_FILE}
done
done
done