#!/bin/bash
# USER='minhphu'
# ROOT=/deep/group/${USER}
# TEMP=${ROOT}/dump

# cp /deep/group/CheXpert/final_ckpts/CheXpert-Ignore/best.pth.tar $TEMP
# cp /deep/group/CheXpert/final_ckpts/CheXpert-Ignore/args.json $TEMP
# cd ${ROOT}/aihc-winter19-robustness/chexpert-model/
python test.py --dataset chexpert \
               --together True \
               --test_csv /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/fine_tune_train__lastFC_epoch20/deploy/valid_with_fake_name/valid.csv \
               --ckpt_path /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/fine_tune_train__lastFC_epoch20/best.pth.tar \
               --phase train \
               --save_dir /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/fine_tune_train__lastFC_epoch20/deploy/ \
               --moco False \

               
