#!/bin/bash
# # 
# # Note:
# # (1) add the deployed CSV file directly to the data/custom_dataset.py on line 44
# # (2) there is no way to add output folder, by default the "results" dir will be saved under the checkpoint dir
# #     for example: /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/fine_tune_train__lastFC_epoch20/results/
#forward-test,backward-test,joint-validation,validation,open_RI,COVID_19_NY_SBU,COVID_19_AR,MIDRC_RICORD_1C,backward-train,forward-train 
echo Beginning testing
SPLIT=equal

MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/1_step_all_CR_stratified
# EXPERIMENT_NAME=COVID_positive_only
# tasks=Yes,No
EXPERIMENT_NAME=CHEXPERT_RESNET
# tasks=CR,DX,F,M,Yes,No,Black_or_African_American,White
# EXPERIMENT_NAME=DEBUG
tasks=F,M,CR,DX
for RAND in 0
do
for RANDOM_STATE in 0
do
for STEP in 0
do
for FINETUNE in full
do
echo ====== RAND ${RAND} == RANDOM STATE ${RANDOM_STATE} == STEP ${STEP} ====== 
python ../test.py --dataset custom \
               --together False \
               --ckpt_path ${MAIN_DIR}/RAND_${RAND}/${EXPERIMENT_NAME}_${RANDOM_STATE}__step_${STEP}/best.pth.tar \
               --phase test \
               --moco False \
               --gpu_ids 3 \
               --custom_tasks ${tasks} \
               --num_workers 70 \
               --eval_folder /gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3/independent_test_sets/ \
               --eval_datasets open_RI \
               --by_patient True
done
done
done
done          
echo All testing complete!