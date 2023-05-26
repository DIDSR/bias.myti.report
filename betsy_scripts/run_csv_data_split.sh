#!/bin/bash
#split the dataset evenly for each subgroup (sex, race and COVID)
#take a csv file as input and output 2 csv files that contain splitted datasets
#can use this script to split dataset in bias amplification indirect approachs (#1b)

#NOTES: currently automatically split based on sex, race and COVID
INPUT_FILE="train.csv"
OUTOUT_1="train_1_baseline.csv"
OUTOUT_2="train_2_baseline.csv"
for BATCH in 0 1 2 3 4
do
for RAND in 0 1 2 3 4
do
IN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run_2b/batch_${BATCH}/RAND_${RAND}
SAVE_DIR=/scratch/yuhang.zhang/OUT/latent_space_run_2b/batch_${BATCH}/RAND_${RAND}
python ../src/csv_data_split.py --in_dir ${IN_DIR} \
                             --save_dir ${SAVE_DIR} \
                             --input_file ${INPUT_FILE} \
                             --output_1 ${OUTOUT_1} \
                             --output_2 ${OUTOUT_2}
done
done