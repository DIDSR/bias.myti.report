#!/bin/bash
# # NOTES
# # LRcustom1: 
# #     declare -a step_LR_start=(1e-4 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5)
# #     declare -a step_LR_decay_step=(1 1 1 1 1 1 1)
# #     declare -a step_LR_decay_gamma=(0.75 0.85 0.85 0.85 0.85 0.85 0.85 0.85)
# # Epcustom1:
# #     declare -a step_epochs=(5 2 2 2 2 2 2)
# #
# # LRcustom2: 
# #     declare -a step_LR_start=(1e-4 1e-5 1e-5 1e-5 1e-5 1e-5 1e-5)
# #     declare -a step_LR_decay_step=(5 1 1 1 1 1 1)
# #     declare -a step_LR_decay_gamma=(0.75 0.85 0.85 0.85 0.85 0.85 0.85 0.85)
# # Epcustom2:
# #     declare -a step_epochs=(25 5 5 5 5 5 5)
# #
# # LRcustom3: 
# #     declare -a step_LR_start=(1e-4 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
# #     declare -a step_LR_decay_step=(5 5 5 5 5 5 5)
# #     declare -a step_LR_decay_gamma=(0.75 0.85 0.85 0.85 0.85 0.85 0.85 0.85)
# # Epcustom3:
# #     declare -a step_epochs=(25 25 25 25 25 25 25)
# #     
# # How to calculate iters
# # step 0 has 15,959 samples, to save every 5 epochs, 
# #     iters_per_save, iters_per_eval, iters_per_visual = 5 * 15959 = 79,795
# #     iters_per_print = 15959 / 10 = 1595, prints 10 iterations/epoch
# # step 1 has 6821 samples, to save every 1 epoch,
# #     iters_per_save, iters_per_eval, iters_per_visual = 1 * 6821 = 6821
# #     iters_per_print = 6821 / 10 = 682, prints 10 iterations/epoch
echo
echo "Running finetune on uignore"
cd "$(dirname "$0")"
MAIN_DIR=/gpfs_projects/ravi.samala/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3
REPO=open_A1

GPU_ID=5
declare -a step_LR_start=(1e-4 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
declare -a step_LR_decay_step=(5 5 5 5 5 5 5)
declare -a step_LR_decay_gamma=(0.75 0.85 0.85 0.85 0.85 0.85 0.85 0.85)
declare -a step_epochs=(25 25 25 25 25 25 25)
# #
declare -a step_iters_per_eval=(79824 6864 6864 6864 6864 6864 6864)
declare -a step_iters_per_print=(1632 720 720 720 720 720 720)
FINETUNE=full
SPLIT=custom
BASE_WEIGHTS=CheXpert # Changes which weights are used for step 0. options CheXpert (specific ckpt specified in train.py), ImageNet, (WIP: Random, MIMIC)

# Change the experiment name to differentiate between different settings (aside from step #) in the same partition folder
# EXPERIMENT_NAME=${BASE_WEIGHTS}_LR${LR_start}_${LR_decay_step}_${LR_decay_gamma}_E${step_epochs[0]}
EXPERIMENT_NAME=${BASE_WEIGHTS}_LRcustom3_Epcustom3
# last layer: module.fc.weight,module.fc.bias
for RAND in 2
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
                   --iters_per_print ${step_iters_per_print[$STEP]} \
                   --iters_per_visual  ${step_iters_per_eval[$STEP]} \
                   --iters_per_eval=${step_iters_per_eval[$STEP]} \
                   --iters_per_save=${step_iters_per_eval[$STEP]} \
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
                   --lr ${step_LR_start[$STEP]} \
                   --lr_scheduler step \
                   --lr_decay_step ${step_LR_decay_step[$STEP]} \
                   --lr_decay_gamma ${step_LR_decay_gamma[$STEP]}
done 
done
done
done
echo "Done!"
