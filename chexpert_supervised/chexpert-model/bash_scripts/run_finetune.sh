#!/bin/bash
echo "Running finetune on uignore"

python ../train.py --ckpt_path /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/checkpoint_0019.pth.tar \
                   --dataset custom \
                   --train_custom_csv /gpfs_projects/ravi.samala/OUT/moco/reorg_chexpert/moving_logs/fine_tune_train_log_small.csv \
                   --val_custom_csv /gpfs_projects/ravi.samala/OUT/moco/reorg_chexpert/moving_logs/valid_log.csv \
                   --save_dir /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/ \
                   --experiment_name fine_tune_train__lastFC_epoch20 \
                   --batch_size 48 \
                   --iters_per_print 48 \
                   --iters_per_visual 48000 \
                   --iters_per_eval=4800 \
                   --iters_per_save=4800 \
                   --gpu_ids 0 \
                   --num_epochs=20 \
                   --metric_name chexpert-competition-AUROC \
                   --maximize_metric True \
                   --scale 320 \
                   --max_ckpts 10 \
                   --keep_topk True \
                   --model ResNet18 \
                   --fine_tuning module.fc.weight,module.fc.bias

echo "Done!"
