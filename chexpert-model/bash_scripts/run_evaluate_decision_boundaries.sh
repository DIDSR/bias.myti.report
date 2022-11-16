#!/bin/bash
MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/1_step_all_CR_stratified_ind_test

for RAND in 0
do
python ../evaluate_decision_boundaries.py --dataset custom \
                                          --together True \
                                          --ckpt_path ${MAIN_DIR}/RAND_${RAND}/CHEXPERT_RESNET_0__step_0/best.pth.tar \
                                          --moco False \
                                          --gpu_ids 7 \
                                          --phase test \
                                          --num_workers 70 \
                                          --custom_tasks=Yes,No \
                                          --test_csv ${MAIN_DIR}/RAND_${RAND}/step_0_validation.csv
done
