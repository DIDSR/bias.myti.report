#!/bin/bash

#shell script to run calibrated equalized odds post-processing bias mitigation algorithm
#takes two csv files that contains output score as inputs. TRAIN_FILE used to train the algorithm, DEPLOY_FILE is the data to deploy
#ouput is the csv file contain output scores after mitigation

#different frozen layers during bias amplification (see 2023 RSNA submission)
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
declare -a CONSTRAINT_ARRAY=("weighted") #cost constraint during mitgation, can be "fnr", "fpr" or "weighted"
CONSTRAIN_WEIGHT=0.5 #weight for fp_rate when "weighted", weight for fn_rate is always 1
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
TRAIN_FILE=by_patient_predictions_val.csv
DEPLOY_FILE=by_patient_predictions.csv
OUTPUT_FILE=cali_eq_odds_${DEPLOY_FILE}
for BATCH in 5
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run/batch_${BATCH}
DEPLOY_INFO_FILE=${MAIN_DIR}/validation_1_image.csv
for LAYER in ${LAYER_ARRAY[@]}
do
for RAND in 0 1 2 3 4
do
TRAIN_INFO_FILE=${MAIN_DIR}/RAND_${RAND}/validation.csv
for RD in 0 1 2 3 4 5 6 7 8 9
do
SUB_DIR=${MAIN_DIR}/RAND_${RAND}/target_model_${LAYER}_RD_${RD}/${EXPERIMENT_NAME}/results/
for CONSTRAINT in ${CONSTRAINT_ARRAY[@]}
do
python ../eq_odds/RSNA_calib_eq_odds.py --test_dir ${SUB_DIR} \
                           --constraints ${CONSTRAINT} \
                           --constraint_weight ${CONSTRAIN_WEIGHT} \
                           --train_file ${TRAIN_FILE} \
                           --deploy_file ${DEPLOY_FILE} \
                           --train_info ${TRAIN_INFO_FILE} \
                           --deploy_info ${DEPLOY_INFO_FILE} \
                           --output_file ${OUTPUT_FILE}
done
done
done
done
done