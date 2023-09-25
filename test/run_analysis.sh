#!/bin/bash
#declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
declare -a LAYER_ARRAY=("50FP")
TESTING_FILE=results__.tsv
VALIDATION_FILE=results__last.tsv
MITIGATION=reject_object_class
for BATCH in 0
do
TESTING_INFO_FILE=/scratch/yuhang.zhang/OUT/latent_space_run_2a_addition_2/batch_${BATCH}/validation_1_image.csv
for LAYER in ${LAYER_ARRAY[@]}
do
EXP_NAME=dcnn_model_${LAYER}
for RAND in 0
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run_2a_addition_2/batch_${BATCH}/RAND_${RAND}
VALIDATION_INFO_FILE=${MAIN_DIR}/validation_${LAYER}.csv
python ../src/myanalysis.py --main_dir ${MAIN_DIR} \
                                --exp_name ${EXP_NAME} \
                                --validation_file ${VALIDATION_FILE} \
                                --testing_file ${TESTING_FILE} \
                                --validation_info_file ${VALIDATION_INFO_FILE} \
                                --testing_info_file ${TESTING_INFO_FILE} \
                                --test_subgroup F M \
                                --post_bias_mitigation ${MITIGATION}
done
done
done