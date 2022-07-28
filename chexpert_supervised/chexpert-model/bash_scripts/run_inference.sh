#!/bin/bash
# # 
# # Note:
# # (1) add the deployed CSV file directly to the data/custom_dataset.py on line 44
# # (2) there is no way to add output folder, by default the "results" dir will be saved under the checkpoint dir
# #     for example: /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/fine_tune_train__lastFC_epoch20/results/
python test.py --dataset custom \
               --together True \
               --ckpt_path "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/updated_fine_tune_train__full_50_epochs/iter_33600.pth.tar" \
               --phase valid \
               --moco False \
               --inference_only \
               --gpu_ids 2

               
