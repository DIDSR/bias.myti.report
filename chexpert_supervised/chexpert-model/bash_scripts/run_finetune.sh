#!/bin/bash
echo "Running finetune on uignore"
cd "$(dirname "$0")"
RAND=0
REPO=MIDRC_RICORD_1C
# last layer: module.fc.weight,module.fc.bias
python ../train.py --ckpt_path /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/checkpoint_0019.pth.tar \
                   --dataset custom \
                   --train_custom_csv /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/RAND_${RAND}/0__20220801_summary_table__${REPO}.csv \
                   --val_custom_csv /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/RAND_${RAND}/1__20220801_summary_table__${REPO}.csv \
                   --save_dir /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/RAND_${RAND} \
                   --experiment_name full_${REPO} \
                   --batch_size 48 \
                   --iters_per_print 48 \
                   --iters_per_visual 48000 \
                   --iters_per_eval=4800 \
                   --iters_per_save=4800 \
                   --gpu_ids 2 \
                   --num_epochs 50 \
                   --metric_name custom-AUROC \
                   --maximize_metric True \
                   --scale 320 \
                   --max_ckpts 10 \
                   --keep_topk True \
                   --model ResNet18 \
                   --fine_tuning full \
                   --custom_tasks custom-tasks \
                   --lr 1e-4

echo "Done!"
