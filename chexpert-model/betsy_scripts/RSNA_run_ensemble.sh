#!/bin/bash
#shell script to ensemble results
#input csv files with output scores from single model
#output csv files contains the mean score from ensembled models

#NOTE: default # of ensembling models is 10, can be changed during input argument

#use this script for bias amplification with both indirect/direct approaches (#1a, 1b, 1c, 2a, 2b)

#different frozen layers during bias amplification (see 2023 RSNA submission)
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct")
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
PRED_FILE=${EXPERIMENT_NAME}/results/by_patient_predictions.csv
OUTPUT_FILE=${EXPERIMENT_NAME}/results/ensemble_by_patient_predictions.csv
for BATCH in 0
do
for LAYER in ${LAYER_ARRAY[@]}
do
for RAND in 0 1 2 3 4
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}/RAND_${RAND}
FOLDER_NAME=target_model_${LAYER}_RD
python ../RSNA_ensemble.py --main_dir ${MAIN_DIR} \
                      --folder_name ${FOLDER_NAME} \
                      --prediction_file ${PRED_FILE} \
                      --output_file ${OUTPUT_FILE}
done
done
done