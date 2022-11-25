#!/bin/bash
EXE="/udsk11/rsamala/git/reproducibility_empirical/src/myinference.py"
IN_DIR="/nas/unas25/rsamala/2022_MAM_CADx_DCNN/RAND_sampling_experiments"
WEIGHT_ITER="checkpoint__249.pth.tar"
# #
# #
MASTER_LOG="${IN_DIR}/master_log_inference.log"
# for ARCH in "googlenet" "resnet18" "wide_resnet50_2" "densenet121"
for ARCH in "googlenet"
do
    # for RAND in 0 1 2 3 4 5 6 7 8 9
    for RAND in 0
    do
        # for FOLD in 0 1 2 3
        for FOLD in 0 1 2 3
        do
            IN_INFERENCE_LIST_FILE="${IN_DIR}/R${RAND}/f${FOLD}/ts.lis"
            IN_DIR2="${IN_DIR}/R${RAND}/f${FOLD}/${ARCH}/"
            WT_FILE="${IN_DIR2}/${WEIGHT_ITER}"
            # # Start the program
            CUDA_LAUNCH_BLOCKING=1 python $EXE -i $IN_INFERENCE_LIST_FILE -w $WT_FILE -d $ARCH -l $MASTER_LOG
        done
    done
done