#!/bin/bash
echo
echo "Running finetune on uignore"
cd "$(dirname "$0")"
MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3
REPO=open_A1

GPU_ID=4
# EPOCHS=500
declare -a step_epochs=(500 50 50 50 50 50 50) # number of epochs for each step of the model training, in order
FINETUNE=full
SPLIT=custom
BASE_WEIGHTS=CheXpert # Changes which weights are used for step 0. options CheXpert (specific ckpt specified in train.py), ImageNet, (WIP: Random, MIMIC)

# Change the experiment name to differentiate between different settings (aside from step #) in the same partition folder
EXPERIMENT_NAME=${BASE_WEIGHTS}_1
# EXPERIMENT_NAME=test_1
# last layer: module.fc.weight,module.fc.bias
for RAND in 0
do 
for OPTION in 0 
do
for STEP in 0 1 2 3 4 5 6
do 
for FINETUNE in full
do
python ../train.py --ckpt_path ${BASE_WEIGHTS} \
                   --dataset custom \
                   --train_custom_csv ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}_${SPLIT}_${#step_epochs[@]}_steps/step_${STEP}.csv \
                   --val_custom_csv ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}_${SPLIT}_${#step_epochs[@]}_steps/step_${STEP}_validation.csv \
                   --save_dir ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}_${SPLIT}_${#step_epochs[@]}_steps \
                   --experiment_name ${EXPERIMENT_NAME}__step_${STEP} \
                   --batch_size 48 \
                   --iters_per_print 480 \
                   --iters_per_visual 48000 \
                   --iters_per_eval=48000 \
                   --iters_per_save=48000 \
                   --gpu_ids ${GPU_ID} \
                   --num_epochs ${step_epochs[$STEP]} \
                   --metric_name custom-AUROC \
                   --maximize_metric True \
                   --scale 320 \
                   --max_ckpts 10 \
                   --keep_topk True \
                   --model ResNet18 \
                   --fine_tuning ${FINETUNE} \
                   --custom_tasks custom-tasks \
                   --lr 1e-4
done 
done
done
done
echo "Done!"
