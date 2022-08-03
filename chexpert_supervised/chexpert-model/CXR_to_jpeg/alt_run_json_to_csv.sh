#!/bin/bash


for RAND in 0 1 2 3 4 5 6 7 8 9
do
    python json_to_csv.py --input /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/RAND_${RAND}/1__20220801_summary_table__COVID_19_AR__20220801_summary_table__COVID_19_NY_SBU.json \
                          --jpeg_loc /gpfs_projects/alexis.burgon/OUT/2022_CXR/
    python json_to_csv.py --input /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/RAND_${RAND}/0__20220801_summary_table__COVID_19_AR__20220801_summary_table__COVID_19_NY_SBU.json \
                          --jpeg_loc /gpfs_projects/alexis.burgon/OUT/2022_CXR/
    
done