#!/bin/bash
# # running with the following venv
# # source /gpfs_projects/ravi.samala/venvs/venv_Py310/bin/activate
# #
python ../src/mytrain.py -i /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/latent_space/batch_1/RAND_0/train.csv \
                         -v /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/latent_space/batch_1/RAND_0/validation.csv \
                         -o /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/latent_space/batch_1/RAND_0/dcnn_test_last_17/ \
                         -d CheXpert_Resnet \
                         -l /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/latent_space/batch_1/RAND_0/dcnn_test_last_17/run_log.log \
                         -p adam \
                         -b 48 \
                         -g 6 \
                         -t 8 \
                         -e 1 \
                         -n 15 \
                         -r 5e-5 \
                         --decay_multiplier 0.5 \
                         --decay_every_N_epoch 5 \
                         --bsave_valid_results_at_epochs True