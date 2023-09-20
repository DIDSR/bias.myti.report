#!/bin/bash
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
#declare -a LAYER_ARRAY=("0BP" "10BP" "25BP" "50BP" "75BP" "90BP" "100BP")
declare -a WEIGHT_ARRAY=("0.5" "0.6" "0.7" "0.8" "0.9" "1" "1.1" "1.2" "1.3" "1.4" "1.5")
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
TEST_SUBGROUP="sex"
#PRED_FILE=ensemble_by_patient_predictions.csv
#PRED_FILE=ensemble_calibrated_predictions.csv
for WEIGHT in ${WEIGHT_ARRAY[@]}
do
PRED_FILE=cali_eq_odds_${WEIGHT}_ensemble_calibrated_predictions.csv
for BATCH in 0 1 2 4
do
for RAND in 0
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run/batch_${BATCH}
INFO_FILE=${MAIN_DIR}/validation_1_image.csv
for RD in 0
do
for LAYER in ${LAYER_ARRAY[@]}
do
SUB_DIR=${MAIN_DIR}/RAND_${RAND}/target_model_${LAYER}_RD_${RD}/${EXPERIMENT_NAME}/
PRED_DIR=${SUB_DIR}/results/
python ../RSNA_subgroup_analysis.py  --rand ${RAND} \
                                --sub_dir ${SUB_DIR} \
                                --pred_dir ${PRED_DIR} \
                                --test_subgroup ${TEST_SUBGROUP} \
                                --prediction_file ${PRED_FILE} \
                                --info_file ${INFO_FILE} \
                                --post_processed True
done
done
done
done
done