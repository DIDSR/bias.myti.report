#!/bin/bash
source /scratch/ravi.samala/anaconda3/envs/venv_python38_2022116/bin/activate

MAIN_DIR=/scratch/alexis.burgon/2022_CXR/model_runs/DB_ensemble/attempt_1
CODE_DIR=/home/alexis.burgon/code/2022_CXR/continual_learning_evaluation/chexpert-model

for RAND in 8 9 10 11 12 13 14 15 16 17 18 19
do
echo RAND $RAND
mkdir ${MAIN_DIR}/RAND_${RAND}/Chex_Res_0__step_0/results
python ${CODE_DIR}/simple_inference.py --dataset custom \
                        --together True \
                        --ckpt_path ${MAIN_DIR}/RAND_${RAND}/Chex_Res_0__step_0/best.pth.tar \
                        --test_csv ${MAIN_DIR}/RAND_${RAND}/validation.csv \
                        --code_dir $CODE_DIR/betsy_scripts \
                        --phase test \
                        --moco False \
                        --gpu_ids 0 \
                        --custom_tasks Yes \
                        --num_workers 10 \
                        --prediction_save_file ${MAIN_DIR}/RAND_${RAND}/Chex_Res_0__step_0/results/validation_predictions.csv
done