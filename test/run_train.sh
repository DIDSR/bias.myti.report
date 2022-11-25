#!/bin/bash
EXE="/udsk11/rsamala/git/reproducibility_empirical/src/mytrain.py"
# IN_TR_LIST_FILE="/udsk11/rsamala/Lists/DCNN_lists/FeatEnggSplit/atm3/CADx_Tr_MAM_256x256__322USF_585DM_1032CBIS_1283SFM__T3222.lis"
# IN_VD_LIST_FILE="/udsk11/rsamala/Lists/DCNN_lists/FeatEnggSplit/atm3/CADx_Tr_MAM_256x256__322USF_585DM_1032CBIS_1283SFM__T3222.lis"
# OUT_DIR="/nas/unas25/rsamala/2022_MAM_CADx_DCNN/alexnet/"
IN_DIR="/nas/unas25/rsamala/2022_MAM_CADx_DCNN/RAND_sampling_experiments"
# #
NUM_EPOCHS=250
BATCH_SIZE=64
# #
OPTIMIZER="sgd"
START_LR=0.001
# #
NUM_THREADS=8
SAVE_EVERY_N_EPOCHS=25
# #
MASTER_LOG="${IN_DIR}/master_log.log"
# for ARCH in "googlenet" "resnet18" "wide_resnet50_2" "densenet121"
for ARCH in "googlenet" "resnet18" "wide_resnet50_2" "densenet121"
do
    # for RAND in 0 1 2 3 4 5 6 7 8 9
    for RAND in 0 1 2 3 4 5 6 7 8 9
    do
        # for FOLD in 0 1 2 3
        for FOLD in 0 1 2 3
        do
            IN_TR_LIST_FILE="${IN_DIR}/R${RAND}/f${FOLD}/tr.lis"
            IN_VD_LIST_FILE="${IN_DIR}/R${RAND}/f${FOLD}/ts.lis"
            OUT_DIR="${IN_DIR}/R${RAND}/f${FOLD}/${ARCH}/"
            mkdir -p -- "$OUT_DIR"
            # # Start the program
            CUDA_LAUNCH_BLOCKING=1 python $EXE -i $IN_TR_LIST_FILE -v $IN_VD_LIST_FILE -o $OUT_DIR -d $ARCH -l $MASTER_LOG -n $NUM_EPOCHS -r $START_LR -b $BATCH_SIZE -p $OPTIMIZER -t $NUM_THREADS -e $SAVE_EVERY_N_EPOCHS
        done
    done
done