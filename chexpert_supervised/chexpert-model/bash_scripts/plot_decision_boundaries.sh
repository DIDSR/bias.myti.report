#!/bin/bash
# #
python test_decision_boundaries.py --dataset custom \
                                    --together True \
                                    --ckpt_path "/gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/RAND_1/full_MIDRC_RICORD_1C/best.pth.tar" \
                                    --phase valid \
                                    --moco False \
                                    --inference_only \
                                    --gpu_ids 3