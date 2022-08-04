#!/bin/bash
CSV_tr="tr__20220801_summary_table__MIDRC_RICORD_1C.json"
CSV_ts="ts__20220801_summary_table__MIDRC_RICORD_1C.json"
for RAND in 10 11 12 13 14
do
    python json_to_csv.py --input /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/RAND_${RAND}/$CSV_tr \
                        --jpeg_loc /gpfs_projects/alexis.burgon/OUT/2022_CXR/MIDRC_RICORD_1C_jpegs \
                        --csv_loc /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/RAND_${RAND}/

    python json_to_csv.py --input /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/RAND_${RAND}/$CSV_ts \
                        --jpeg_loc /gpfs_projects/alexis.burgon/OUT/2022_CXR/MIDRC_RICORD_1C_jpegs \
                        --csv_loc /gpfs_projects/ravi.samala/OUT/2022_CXR/SPIE2023_runs/atm2/RAND_${RAND}/
done 


# python json_to_csv.py --input /gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_NY_SBU.json \
#                         --jpeg_loc /gpfs_projects/alexis.burgon/OUT/2022_CXR/COVID_19_NY_SBU_jpegs \
#                         --csv_loc /gpfs_projects/ravi.samala/OUT/2022_CXR/202208/