#!/bin/bash
# #
python test_decision_boundaries.py --dataset custom \
                                    --together True \
                                    --ckpt_path /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/fine_tune_train__lastFC_epoch20/best.pth.tar \
                                    --phase valid \
                                    --moco False \
                                    --inference_only \
                                    --gpu_ids 2