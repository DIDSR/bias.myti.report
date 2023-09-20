#!/bin/bash
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
#declare -a LAYER_ARRAY=("0FP" "10FP" "25FP" "50FP" "75FP" "90FP" "100FP")
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
TEST_SUBGROUP="sex" #choose the subgroup to be mitigated, currently support sex and race
PRED_FILE=calibrated_validation_predictions.csv #calibrated validation prediciton file for training
PROC_FILE=ensemble_calibrated_predictions.csv #calibrated testing prediction file for mitigation
for BATCH in 0 1 2 3 4
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run/batch_${BATCH}
INFO_FILE_2=${MAIN_DIR}/validation_1_image.csv
for RAND in 0
do
INFO_FILE=${MAIN_DIR}/RAND_${RAND}/validation.csv
for RD in 0
do
for LAYER in ${LAYER_ARRAY[@]}
do
SUB_DIR=${MAIN_DIR}/RAND_${RAND}/target_model_${LAYER}_RD_${RD}/${EXPERIMENT_NAME}/
PRED_DIR=${SUB_DIR}/results/
python ../reject_option_classification.py  --rand ${RAND} \
                                --sub_dir ${SUB_DIR} \
                                --pred_dir ${PRED_DIR} \
                                --test_subgroup ${TEST_SUBGROUP} \
                                --prediction_file ${PRED_FILE} \
                                --process_file ${PROC_FILE} \
                                --info_file ${INFO_FILE} \
                                --info_file_2 ${INFO_FILE_2} \
                                --post_processed False
done
done
done
done