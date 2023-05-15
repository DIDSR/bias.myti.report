#!/bin/bash
# # running with the following venv
# # source /gpfs_projects/ravi.samala/venvs/venv_Py310/bin/activate
# #
python ../src/mytrain.py -i /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/subgroup_size/50w50b/RAND_0/train.csv -v /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/subgroup_size/50w50b/RAND_0/train.csv -o /gpfs_projects/ravi.samala/OUT/2022_CXR/temp/ -d CheXpert_Resnet -l /gpfs_projects/ravi.samala/OUT/2022_CXR/temp/ -p adam -g 0 -t 8 -e 5 -n 25
