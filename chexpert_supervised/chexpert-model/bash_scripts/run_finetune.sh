#!/bin/bash
echo "Running finetune on uignore"
cd "$(dirname "$0")"
python ../train.py --ckpt_path /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/checkpoint_0019.pth.tar \
                   --dataset custom \
                   --train_custom_csv /gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/TCIA_1C_train.csv \
                   --val_custom_csv /gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/TCIA_1C_valid.csv \
                   --save_dir /gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/ \
                   --experiment_name updated_fine_tune_train__full_50_epochs_change_max \
                   --batch_size 48 \
                   --iters_per_print 48 \
                   --iters_per_visual 48000 \
                   --iters_per_eval=4800 \
                   --iters_per_save=4800 \
                   --gpu_ids 2 \
                   --num_epochs=50 \
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
