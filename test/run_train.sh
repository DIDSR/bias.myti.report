#!/bin/bash
EXE="/udsk11/rsamala/git/reproducibility_empirical/src/mytrain.py"
IN_TR_LIST_FILE="/udsk11/rsamala/Lists/DCNN_lists/FeatEnggSplit/atm3/CADx_Tr_MAM_256x256__322USF_585DM_1032CBIS_1283SFM__T3222.lis"
IN_VD_LIST_FILE="/udsk11/rsamala/Lists/DCNN_lists/FeatEnggSplit/atm3/CADx_Tr_MAM_256x256__322USF_585DM_1032CBIS_1283SFM__T3222.lis"
OUT_DIR="/nas/unas25/rsamala/2022_MAM_CADx_DCNN/alexnet/"
# #
NUM_EPOCHS=250
BATCH_SIZE=64
# #
ARCH="alexnet"     # # "googlenet", "resnet18", "wide_resnet50_2", "densenet121"
OPTIMIZER="sgd"
START_LR=0.001
# #
NUM_THREADS=8
# #
# # Start the program
CUDA_LAUNCH_BLOCKING=1 python $EXE -i $IN_TR_LIST_FILE -v $IN_VD_LIST_FILE -o $OUT_DIR -d $ARCH -l $OUT_DIR -n $NUM_EPOCHS -r $START_LR -b $BATCH_SIZE -p $OPTIMIZER -t $NUM_THREADS