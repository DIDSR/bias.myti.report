#!/bin/bash
declare -a LAYER_ARRAY=("last_17")
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
PRED_FILE=${EXPERIMENT_NAME}/results/by_patient_predictions.csv
OUTPUT_FILE=${EXPERIMENT_NAME}/results/ensemble_by_patient_predictions.csv
for BATCH in 5
do
for LAYER in ${LAYER_ARRAY[@]}
do
for RAND in 0
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run/batch_${BATCH}/RAND_${RAND}
FOLDER_NAME=target_model_${LAYER}_RD
python ../RSNA_ensemble.py --main_dir ${MAIN_DIR} \
                      --folder_name ${FOLDER_NAME} \
                      --prediction_file ${PRED_FILE} \
                      --output_file ${OUTPUT_FILE}
done
done
done