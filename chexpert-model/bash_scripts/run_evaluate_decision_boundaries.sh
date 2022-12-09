#!/bin/bash
MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v4/1_step_all_CR_stratified_ind_test

for RAND in 0
do
for VAL_SUB in 100_equal
do
python ../evaluate_decision_boundaries.py --dataset custom \
                                          --together True \
                                          --ckpt_path ${MAIN_DIR}/RAND_${RAND}/DEBUG__step_0/${VAL_SUB}__best.pth.tar \
                                          --moco False \
                                          --gpu_ids 3 \
                                          --phase test \
                                          --num_workers 70 \
                                          --custom_tasks=Yes,No \
                                          --test_csv ${MAIN_DIR}/RAND_${RAND}/step_0__${VAL_SUB}_validation.csv
done
done
