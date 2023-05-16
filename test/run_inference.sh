#!/bin/bash
# # running with the following venv
# # source /gpfs_projects/ravi.samala/venvs/venv_Py310/bin/activate
# #
python ../src/myinference.py -i /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/subgroup_size/50w50b/RAND_0/train.csv -w /gpfs_projects/ravi.samala/OUT/2022_CXR/temp/pytorch_last_epoch_model.onnx -d CheXpert_Resnet -l /gpfs_projects/ravi.samala/OUT/2022_CXR/temp/inference_log.log
