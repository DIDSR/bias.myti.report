#!/bin/bash

# INPUTS:
    # MAIN_DIR = location of all of the random seeds that are to be used
    # EXPERIMENT_NAME = experiment name given during run_finetune (model folder)
    # -- test_csv = csv to evaluate
    # -- prediction save file = output file name



MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/DB_ensemble_BETSY/attempt_1/
EXPERIMENT_NAME=Chex_Res_0__step_0

for RAND in 0 
do
python ../simple_inference.py --dataset custom \
               --together False \
               --ckpt_path ${MAIN_DIR}/RAND_${RAND}/${EXPERIMENT_NAME}/best.pth.tar \
               --phase test \
               --moco False \
               --gpu_ids 3 \
               --custom_tasks Yes \
               --num_workers 70 \
               --test_csv "/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/DB_ensemble_BETSY/attempt_1/independent_tests/open_R1_groundtruth.csv" \
               --by_patient False \
               --prediction_save_file ${MAIN_DIR}/independent_tests/open_R1_predictions_RAND_${RAND}.csv
done