#!/bin/bash
# #
python test_decision_boundaries.py --dataset custom \
                                    --together True \
                                    --ckpt_path "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/updated_fine_tune_train__full_50_epochs/iter_33600.pth.tar" \
                                    --phase valid \
                                    --moco False \
                                    --inference_only \
                                    --gpu_ids 3