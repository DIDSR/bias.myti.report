#!/bin/bash
#shell script to run subgroup bias measurements computation
#input a csv prediction file contains output score, and a csv info file contains patient attribute info
#output 2 csv files contain nuanced measurements and other measurements
#current support nuanced AUCs (subgroup-background AUROC, AEGs),
#demographic parity, TPR/TNR, subgroup AUROC, subgroup NLL, PPV

#different frozen layers during bias amplification (see 2023 RSNA submission)
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
TEST_SUBGROUP="sex"  #interested subgroup to measure bias, can be "sex", "race", "modality"
PRED_FILE=ensemble_by_patient_predictions.csv
POST_PRCS=False #whether the input file is after bias mitigation
for BATCH in 5
do
for RAND in 0 1 2 3 4
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
                                --post_processed ${POST_PRCS} 
done
done
done
done