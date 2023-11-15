#!/bin/bash
# # perform last layer retraining for bias mitigation
# # running with the following venv
# # source /gpfs_projects/ravi.samala/venvs/venv_Py310/bin/activate
# #
source /scratch/ravi.samala/venvs/conda_venv/venv_python310_20230608/bin/activate
MAIN_DIR=/scratch/yuhang.zhang/CXR_OUT/latent_space_run_2a_addition
declare -a FRACTION_ARRAY=("0FP")
for BATCH in 0
do
for RAND in 0
do
for FRACTION in ${FRACTION_ARRAY[@]}
do
for RANDOM_STATE in 0
do
MODEL_PATH=${MAIN_DIR}/batch_${BATCH}/RAND_${RAND}/dcnn_model_75subsample_CheXpert_Resnet_${FRACTION}_RD_${RANDOM_STATE}/checkpoint__8.pth.tar
EXP_NAME=last_layer_retrain_model_test
python ../src/last_layer_retrain.py -i ${MAIN_DIR}/batch_${BATCH}/RAND_${RAND}/train_50FP.csv \
                                   -v ${MAIN_DIR}/batch_${BATCH}/RAND_${RAND}/validation.csv \
                                   -o ${MAIN_DIR}/batch_${BATCH}/RAND_${RAND}/${EXP_NAME}_${FRACTION}_RD_${RANDOM_STATE}/ \
                                   -d resnet18 \
                                   -l ${MAIN_DIR}/batch_${BATCH}/RAND_${RAND}/${EXP_NAME}_${FRACTION}_RD_${RANDOM_STATE}/run_log.log \
                                   -f ${MODEL_PATH} \
                                   -p adam \
                                   -b 48 \
                                   -g 0 \
                                   -t 8 \
                                   -e 1 \
                                   -n 12 \
                                   -r 1e-5 \
                                   --decay_multiplier 0.5 \
                                   --decay_every_N_epoch 3 \
                                   --bsave_valid_results_at_epochs True \
                                   --random_state ${RANDOM_STATE} \

done
done
done
done