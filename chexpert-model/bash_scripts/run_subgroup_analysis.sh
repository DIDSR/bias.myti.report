#!/bin/bash
TEST_DIR="/gpfs_projects/yuhang.zhang/OUT/2022_CXR/bias/subgroup_size/25w75b/"
TEST_SUBGROUP="race"
for RAND in 0
do
python ../subgroup_analysis.py  --rand $RAND \
                                --test_dir ${TEST_DIR} \
                                --test_subgroup ${TEST_SUBGROUP}
done