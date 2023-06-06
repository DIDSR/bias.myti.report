#!/bin/bash
source /scratch/ravi.samala/anaconda3/envs/venv_python38_2022116/bin/activate
BASE_WEIGHTS=CheXpert_Resnet
EXPERIMENT_NAME=${BASE_WEIGHTS}_subgroup_size_decay_50_lr_5e-5
#LAYER=last_3
declare -a LAYER_ARRAY=("last_17" "last_6" "last_4" "last_3" "direct")
for BATCH in 0
do
MAIN_DIR=/scratch/yuhang.zhang/OUT/temp/batch_${BATCH}
for LAYER in ${LAYER_ARRAY[@]}
do
for RAND in 0 1 2 3 4
do
echo RAND $RAND
for RD in 0 1 2 3 4 5 6 7 8 9
do
python ../subgroup_test.py          --dataset custom \
                                    --custom_tasks custom-tasks \
                                    --together True \
                                    --ckpt_path ${MAIN_DIR}/RAND_${RAND}/target_model_${LAYER}_RD_${RD}/${EXPERIMENT_NAME}/best.pth.tar \
                                    --phase test \
                                    --moco False \
                                    --gpu_ids 0 \
                                    --eval_folder ${MAIN_DIR}/ \
                                    --save_dir ${MAIN_DIR}/RAND_${RAND}/target_model_${LAYER}_RD_${RD}/${EXPERIMENT_NAME}/testing/ \
                                    --eval_datasets validation \
                                    --by_patient True
done
done
done
done