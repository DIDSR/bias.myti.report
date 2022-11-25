#!/bin/bash
EXE="/udsk11/rsamala/git/reproducibility_empirical/src/utils/generate_kfold_cross_validation_list_files.py"
IN_TR_LIST_FILE="/udsk11/rsamala/Lists/DCNN_lists/FeatEnggSplit/atm3/CADx_Tr_MAM_256x256__322USF_585DM_1032CBIS_1283SFM__T3222.lis"
OUT_DIR="/nas/unas25/rsamala/2022_MAM_CADx_DCNN/RAND_sampling_experiments/R0/"
K_FOLDS=4
# #
# #
# # Start the program
python $EXE -i $IN_TR_LIST_FILE -o $OUT_DIR -f $K_FOLDS