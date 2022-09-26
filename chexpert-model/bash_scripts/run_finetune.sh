#!/bin/bash
echo
echo "Running finetune on uignore"
cd "$(dirname "$0")"
MAIN_DIR=/gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/scenario_1
# RAND=0
# OPTION=0
REPO=MIDRC_RICORD_1C
GPU_ID=3
EPOCHS=500
FINETUNE=full
# STEP=0
SPLIT=equal_acc
# # --save_dir ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION} \
# last layer: module.fc.weight,module.fc.bias
for RAND in 0 1 2 3 4
do 
for OPTION in 0 1
do
for STEP in 0 1 2 3
do 
for FINETUNE in full
do
python ../train.py --ckpt_path /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/checkpoint_0019.pth.tar \
                   --dataset custom \
                   --train_custom_csv ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}_${SPLIT}/step_${STEP}.csv \
                   --val_custom_csv ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}_${SPLIT}/validation.csv \
                   --save_dir ${MAIN_DIR}/RAND_${RAND}_OPTION_${OPTION}_${SPLIT} \
                   --experiment_name ${FINETUNE}_step_${STEP}_${REPO}_${EPOCHS}_epochs \
                   --batch_size 48 \
                   --iters_per_print 48 \
                   --iters_per_visual 48000 \
                   --iters_per_eval=4800 \
                   --iters_per_save=4800 \
                   --gpu_ids ${GPU_ID} \
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
done 
done
done
done
echo "Done!"
