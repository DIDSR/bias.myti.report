#!/bin/bash
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
#declare -a LAYER_ARRAY=("0BP" "10BP" "25BP" "50BP" "75BP" "90BP" "100BP")
declare -a CONSTRAINT_ARRAY=("weighted")
declare -a WEIGHT_ARRAY=("0.5" "0.6" "0.7" "0.8" "0.9" "1" "1.1" "1.2" "1.3" "1.4" "1.5")
TEST_SUBGROUP="sex" #choose the subgroup to be mitigated, currently support sex and race
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
TRAIN_FILE=calibrated_validation_predictions.csv #calibrated validation prediciton file for training
DEPLOY_FILE=ensemble_calibrated_predictions.csv  #calibrated testing prediction file for mitigation
for BATCH in 0
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run/batch_${BATCH}
DEPLOY_INFO_FILE=${MAIN_DIR}/validation_1_image.csv
for LAYER in ${LAYER_ARRAY[@]}
do
for RAND in 0
do
TRAIN_INFO_FILE=${MAIN_DIR}/RAND_${RAND}/validation.csv
for RD in 0
do
SUB_DIR=${MAIN_DIR}/RAND_${RAND}/target_model_${LAYER}_RD_${RD}/${EXPERIMENT_NAME}/results/
for CONSTRAINT in ${CONSTRAINT_ARRAY[@]}
do
for WEIGHT in ${WEIGHT_ARRAY[@]}
do
OUTPUT_FILE=cali_eq_odds_${WEIGHT}_${DEPLOY_FILE}
python ../RSNA_calib_eq_odds.py --test_dir ${SUB_DIR} \
                           --constraints ${CONSTRAINT} \
                           --constraint_weight ${WEIGHT} \
                           --train_file ${TRAIN_FILE} \
                           --deploy_file ${DEPLOY_FILE} \
                           --train_info ${TRAIN_INFO_FILE} \
                           --deploy_info ${DEPLOY_INFO_FILE} \
                           --output_file ${OUTPUT_FILE} \
                           --test_subgroup ${TEST_SUBGROUP}
done
done
done
done
done
done