#!/bin/bash
MAIN_DIR="/gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/subgroup_size/100w0b/"
GPU_ID=4
BASE_WEIGHTS=CheXpert-Mimic_Resnet # Changes which weights are used for step 0. options CheXpert (specific ckpt specified in train.py), ImageNet, (WIP: Random, MIMIC)
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_75_lr_1e-4
for RAND in 9
do 
for FINETUNE in full
do
python ../train.py --ckpt_path ${BASE_WEIGHTS} \
                   --dataset custom \
                   --train_custom_csv ${MAIN_DIR}/RAND_${RAND}/train.csv \
                   --val_custom_csv ${MAIN_DIR}/RAND_${RAND}/validation.csv \
                   --save_dir ${MAIN_DIR}/RAND_${RAND}/output_model \
                   --experiment_name ${EXPERIMENT_NAME} \
                   --batch_size 48 \
                   --iters_per_print 1632 \
                   --iters_per_visual  79824 \
                   --iters_per_eval=6048 \
                   --iters_per_save=6048 \
                   --gpu_ids ${GPU_ID} \
                   --num_epochs 25 \
                   --metric_name custom-AUROC \
                   --maximize_metric True \
                   --scale 320 \
                   --max_ckpts 10 \
                   --keep_topk True \
                   --model ResNet18 \
                   --fine_tuning ${FINETUNE} \
                   --custom_tasks custom-tasks \
                   --lr 1e-4 \
                   --lr_scheduler exponential \
                   --lr_decay_step 5 \
                   --lr_decay_gamma 0.75
done 
done