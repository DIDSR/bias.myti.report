#!/bin/bash
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct" "reverse_last_17" "reverse_last_6" "reverse_last_4" "reverse_last_3")
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_30_lr_5e-5
FILE_NAME=ensemble_weighted_5_cali_eq_odds
FILE_NAME_1=subgroup_fairness_${FILE_NAME}.csv
FILE_NAME_2=subgroup_nuance_auc_${FILE_NAME}.csv
for BATCH in 5
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/latent_space_run/batch_${BATCH}
for RD in 0
do
for LAYER in ${LAYER_ARRAY[@]}
do
FOLDER_NAME=target_model_${LAYER}_RD_${RD}/${EXPERIMENT_NAME}/
python ../RSNA_subgroup_summarize.py --main_dir ${MAIN_DIR} \
                                --file_name ${FILE_NAME} \
                                --folder_name ${FOLDER_NAME} \
                                --file_name_1 ${FILE_NAME_1} \
                                --file_name_2 ${FILE_NAME_2} \
                                --layer ${LAYER}
done
done
done