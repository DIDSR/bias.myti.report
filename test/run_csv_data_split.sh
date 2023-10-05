#!/bin/bash
#subsampling the dataset with the same rate in each subgroup (sex, race and COVID)
#manipulate on diease prevalence in different subgroups. Current supports race and sex.
#take a csv data file as input and output a csv file contains result data 
#can use this script to split dataset in bias amplification direct approachs

#NOTES: diease prevalence is complementary in subgroups. For example, 10% prevalence in female
#correpsonds to 90% prevalence in male.

INPUT_FILE="train.csv"
declare -a FRACTION_ARRAY=('0' '0.1' '0.25' '0.5' '0.75' '0.9' '1') #ranging from 0 to 1
for BATCH in 0
do
for RAND in 0
do
for FRACTION in ${FRACTION_ARRAY[@]}
do
IN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}/RAND_${RAND}
SAVE_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}/RAND_${RAND}
python ../src/csv_data_split.py --prevalence ${FRACTION} \
                                --subsample 0.5 \
                                --test_subgroup Black \
                                --in_dir ${IN_DIR} \
                                --save_dir ${SAVE_DIR} \
                                --input_file ${INPUT_FILE}
done
done
done