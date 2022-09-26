#!/bin/bash
# # 
# # Note:
# # (1) add the deployed CSV file directly to the data/custom_dataset.py on line 44
# # (2) there is no way to add output folder, by default the "results" dir will be saved under the checkpoint dir
# #     for example: /gpfs_projects/ravi.samala/OUT/moco/experiments/ravi.samala/r8w1n416_20220715h15_tr_mocov2_20220715-172742/finetune/fine_tune_train__lastFC_epoch20/results/
echo Beginning testing
SPLIT=equal

for RAND in 0 1
do
for OPTION in 0 1
do
for STEP in 0 1 2 3
do
for FINETUNE in full
do
python ../test.py --dataset custom \
               --together True \
               --ckpt_path /gpfs_projects/alexis.burgon/OUT/2022_CXR/model_runs/scenario_1/RAND_${RAND}_OPTION_${OPTION}_${SPLIT}/${FINETUNE}_step_${STEP}_MIDRC_RICORD_1C_500_epochs/best.pth.tar \
               --phase test \
               --moco False \
               --gpu_ids 2 \
               --eval_folder /gpfs_projects/alexis.burgon/OUT/2022_CXR/data_summarization/20220823 \
               --eval_datasets validation,COVID_19_NY_SBU,COVID_19_AR,open_AI,open_RI
done
done
done
done          
