#!/bin/bash

MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/DB_ensemble_BETSY/attempt_1

for RAND in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
# for RAND in 0
do
echo ========= RAND $RAND =========
python ../calibrate.py --dataset custom \
                        --together True \
                        --ckpt_path ${MAIN_DIR}/RAND_${RAND}/Chex_Res_0__step_0/best.pth.tar \
                        --test_csv ${MAIN_DIR}/RAND_${RAND}/validation.csv \
                        --phase test \
                        --moco False \
                        --gpu_ids 3 \
                        --tasks Yes \
                        --num_workers 70 \
                        --model_uncertainty True \
                        --prediction_file ${MAIN_DIR}/RAND_${RAND}/Chex_Res_0__step_0/results/validation_predictions.csv 
done 