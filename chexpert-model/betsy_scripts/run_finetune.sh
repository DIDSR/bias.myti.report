#!/bin/bash
source /scratch/ravi.samala/anaconda3/envs/venv_python38_2022116/bin/activate
GPU_ID=0
BASE_WEIGHTS=CheXpert_Resnet # Changes which weights are used for step 0. options CheXpert (specific ckpt specified in train.py), ImageNet, (WIP: Random, MIMIC)
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_50_lr_5e-5
#RANDOM_STATE=0
TASK_1=F
TASK_1_R=M
TASK_2=Yes
LAYER=direct
for BATCH in 0
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}
for RAND in 0 1 2 3 4
do
for RANDOM_STATE in 0 1 2 3 4 5 6 7 8 9
do
python ../train.py --ckpt_path ${BASE_WEIGHTS} \
                   --dataset custom \
                   --train_custom_csv ${MAIN_DIR}/RAND_${RAND}/train.csv \
                   --val_custom_csv ${MAIN_DIR}/RAND_${RAND}/validation.csv \
                   --save_dir ${MAIN_DIR}/RAND_${RAND}/target_model_${LAYER}_RD_${RANDOM_STATE} \
                   --experiment_name ${EXPERIMENT_NAME} \
                   --batch_size 48 \
                   --iters_per_print 1632 \
                   --iters_per_visual  79824 \
                   --iters_per_eval=960 \
                   --iters_per_save=960 \
                   --gpu_ids ${GPU_ID} \
                   --num_epochs 15 \
                   --metric_name custom-AUROC \
                   --maximize_metric True \
                   --scale 320 \
                   --max_ckpts 15 \
                   --keep_topk True \
                   --model ResNet18 \
                   --fine_tuning full \
                   --custom_tasks ${TASK_2} \
                   --lr 5e-5 \
                   --lr_scheduler step \
                   --lr_decay_step 5 \
                   --lr_decay_gamma 0.5 \
                   --random_state ${RANDOM_STATE} \
                   --by_patient True
done
done
done