#!/bin/bash
# to run: ./basic_partitioning.py
EXE="../generate_partitions.py"
repo="open_A1"
save_dir="/gpfs_projects/alexis.burgon/OUT/2022_CXR/temp/generate_partitions_test/"
experiment_name="test_batching"

for batch in 1 2
do
    for RAND in 0 1 2 3
    do
        python $EXE -r $RAND \
            --repo $repo \
            --partitions train validation-1 validation-2 test \
            --partition-sizes 0.5 0.1 0.2 0.2 \
            --partition-distributions equal \
            --batch validation-2 test \
            --batch-rand $batch \
            --tasks COVID_positive \
            --save-location $save_dir \
            --experiment-name $experiment_name \
            --summary-attributes sex race 
    done
done