#!/bin/bash
#running environment: on OpenHPC
# source /gpfs_projects/ravi.samala/venvs/venv_Py310/bin/activate
#Currently all the results are directly stored in the python script
#TODO: modify to read in csv files and make plots
SAVE_DIR=/home/yuhang.zhang/figures
python ../bias_results_plot.py     --save_dir ${SAVE_DIR}
