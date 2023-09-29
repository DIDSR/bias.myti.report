#!/bin/bash
# # running with the following venv
# # source /gpfs_projects/ravi.samala/venvs/venv_Py310/bin/activate
# #
python ../src/myinference.py  -i /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/latent_space/batch_1/validation_1_image.csv \
                              -w /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/latent_space/batch_1/RAND_0/dcnn_test_last_17/pytorch_last_epoch_model.onnx \
                              -d CheXpert_Resnet \
                              -b 48 \
                              -g 0 \
                              -l /gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/latent_space/batch_1/RAND_0/dcnn_test_last_17/inference_log.log