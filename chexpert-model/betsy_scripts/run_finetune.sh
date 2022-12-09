#!/bin/bash
# source /home/alexis.burgon/anaconda3/envs/env_moco_cxr_2/bin/activate
source /scratch/ravi.samala/anaconda3/envs/venv_python38_2022116/bin/activate
# source /scratch/ravi.samala/anaconda3/envs/venv_python369/bin/activate
echo
echo "Running finetune on uignore"
cd "$(dirname "$0")"
# MAIN_DIR=/gpfs_projects/ravi.samala/OUT/2022_CXR/model_runs/open_A1_scenario_1_v3
MAIN_DIR=/scratch/alexis.burgon/2022_CXR/model_runs/DB_ensemble/attempt_1
# MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/debug_test
REPO=open_A1

GPU_ID=0
declare -a step_LR_start=(1e-4 5e-5 5e-5 5e-5 5e-5 5e-5 5e-5)
declare -a step_LR_decay_step=(5 5 5 5 5 5 5)
declare -a step_LR_decay_gamma=(0.75 0.85 0.85 0.85 0.85 0.85 0.85 0.85)
declare -a step_epochs=(25 25 25 25 25 25 25)
# #
# declare -a step_iters_per_eval=(79824 6864 6864 6864 6864 6864 6864)
# declare -a step_iters_per_print=(1632 720 720 720 720 720 720)
# declare -a step_iters_per_eval=(144 6864 6864 6864 6864 6864 6864)
# declare -a step_iters_per_print=(144 720 720 720 720 720 720)
# declare -a step_iters_per_eval=(75024)
declare -a step_iters_per_eval=(37536) # approx every 2-3 epochs, lower than normal to account for the val subsets
declare -a step_iters_per_print=(2640)
FINETUNE=full
BASE_WEIGHTS=CheXpert_Resnet # Changes which weights are used for step 0. options CheXpert (specific ckpt specified in train.py), ImageNet, (WIP: Random, MIMIC)

# Change the experiment name to differentiate between different settings (aside from step #) in the same partition folder
# EXPERIMENT_NAME=${BASE_WEIGHTS}_LR${LR_start}_${LR_decay_step}_${LR_decay_gamma}_E${step_epochs[0]}

# last layer: module.fc.weight,module.fc.bias

for RANDOM_STATE in 0 # this indicates the model's random state
do
# EXPERIMENT_NAME=CHEXPERT_RESNET_${RANDOM_STATE}
# EXPERIMENT_NAME=VALIDATION_SIZE_${RANDOM_STATE}
EXPERIMENT_NAME=Chex_Res_${RANDOM_STATE}
# EXPERIMENT_NAME=subgroups__${RANDOM_STATE}
for RAND in 19 # this indicates the data partition
do 
for OPTION in 0 
do
for STEP in 0 
do 
for FINETUNE in full
do
# EXPERIMENT_NAME=${BASE_WEIGHTS}__${RANDOM_STATE}__${FINETUNE}
python /home/alexis.burgon/code/2022_CXR/continual_learning_evaluation/chexpert-model/train.py --ckpt_path ${BASE_WEIGHTS} \
                   --dataset custom \
                   --code_dir /home/alexis.burgon/code/2022_CXR/continual_learning_evaluation/chexpert-model/betsy_scripts \
                   --train_custom_csv ${MAIN_DIR}/RAND_${RAND}/train.csv \
                   --val_custom_csv ${MAIN_DIR}/RAND_${RAND}/validation.csv \
                   --save_dir ${MAIN_DIR}/RAND_${RAND} \
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
                   --num_workers 10 \
                   --scale 320 \
                   --max_ckpts 15 \
                   --keep_topk True \
                   --model ResNet18 \
                   --fine_tuning ${FINETUNE} \
                   --custom_tasks Yes \
                   --lr ${step_LR_start[$STEP]} \
                   --lr_scheduler step \
                   --lr_decay_step ${step_LR_decay_step[$STEP]} \
                   --lr_decay_gamma ${step_LR_decay_gamma[$STEP]} \
                   --random_state ${RANDOM_STATE} \
                   --by_patient True
done 
done
done
done
done
echo "Done!"
