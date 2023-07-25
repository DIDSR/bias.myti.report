#!/bin/bash
# to run: ./segmentation_evolving.py
EXE="../generate_partitions.py"
repo="BraTS"
save_dir=""
experiment_name=""

for batch in 0
do
    for RAND in 0
    do
        python $EXE -r $RAND \
            --repo $repo \
            --steps 2 \
            --step-sizes 1 1 \
            --partitions train validation test \
            --partition-sizes 0.7 0.1 0.2 \
            --batch test \
            --batch-rand $batch \
            --save-location $save_dir \
            --experiment-name $experiment_name
    done
done