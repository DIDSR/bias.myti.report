#!/bin/bash
source /home/alexis.burgon/anaconda3/envs/env_moco_cxr_2/bin/activate
echo
echo "Running finetune on uignore"
cd "$(dirname "$0")"
MAIN_DIR=/scratch/alexis.burgon/2022_CXR/model_runs/20220902
RAND=0
OPTION=0
REPO=MIDRC_RICORD_1C
EPOCHS=500
FINETUNE=last_12

# last layer: module.fc.weight,module.fc.bias
python ../train.py --ckpt_path /scratch/alexis.burgon/2022_CXR/model_runs/moco_checkpoints/checkpoint_0019.pth.tar \
                   --dataset custom \
                   --train_custom_csv ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}/tr__summary_table__${REPO}.csv \
                   --val_custom_csv ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}/ts__summary_table__${REPO}.csv \
                   --save_dir ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION} \
                   --experiment_name ${FINETUNE}_${REPO}_${EPOCHS}_epochs \
                   --batch_size 48 \
                   --iters_per_print 48 \
                   --iters_per_visual 48000 \
                   --iters_per_eval=480 \
                   --iters_per_save=480 \
                   --num_epochs ${EPOCHS} \
                   --metric_name custom-AUROC \
                   --maximize_metric True \
                   --scale 320 \
                   --max_ckpts 10 \
                   --keep_topk True \
                   --model ResNet18 \
                   --fine_tuning ${FINETUNE} \
                   --custom_tasks custom-tasks \
                   --lr 1e-4 

echo "Done!"
