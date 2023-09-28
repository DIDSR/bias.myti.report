#!/bin/bash
#declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
declare -a EXP_ARRAY=("50BP")
INPUT_FILE=CheXpert_Resnet_subgroup_size_decay_50_lr_5e-5/subgroup_fairness_ensemble_by_patient_predictions.csv
OUTPUT_FILE=summary_test.csv
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run_2b_addition/
for EXP in ${EXP_ARRAY[@]}
do
EXP_NAME=target_model_${EXP}
python ../src/mysummary.py     --main_dir ${MAIN_DIR} \
                                --exp_name ${EXP_NAME} \
                                --input_file ${INPUT_FILE} \
                                --output_file ${OUTPUT_FILE} \
                                --test_subgroup Black White \
                                --batch_num 25
done
