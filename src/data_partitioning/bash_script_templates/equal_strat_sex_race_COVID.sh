#!/bin/bash
# to run: ./equal_strat_sex_race_COVID.sh
EXE="../generate_partitions.py"
repo="open_A1"
save_dir=""
experiment_name=""

for RAND in 0
do
    python $EXE -r $RAND \
        --repo $repo \
        --partitions train validation-1 validation-2 test \
        --partition-sizes 0.5 0.1 0.2 0.2 \
        --partition-distributions equal-binary \
        --tasks COVID_positive \
        --save-location $save_dir \
        --experiment-name $experiment_name \
        --attributes sex race 
done