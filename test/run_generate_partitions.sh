#!/bin/bash
# # Program to 
# # to run: >> ./run_main_summarize.sh
# # 
# # 
# #
EXE="../src/generate_partitions.py"
# declare -a IN_summary=("/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__MIDRC_RICORD_1C.json" \
# 						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_AR.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_NY_SBU.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__open_AI.json" \
#						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__open_RI.json")

declare -a IN_summary=("/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__MIDRC_RICORD_1C.json" \
						"/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__COVID_19_AR.json")
# declare -a IN_summary=("/gpfs_projects/ravi.samala/OUT/2022_CXR/202208/20220801_summary_table__open_RI.json")
OUT_dir="/gpfs_projects/ravi.samala/OUT/2022_CXR/partitions/"


# #
# #
# #
# # -----------------------------------------------------------------------
# # -----------------------------------------------------------------------
# # -----------------------------------------------------------------------
# #
# # get length of an array
arraylength=${#IN_summary[@]}
# #
param_IN_summary=""
# # create input params
for (( i=0; i<${arraylength}; i++ ));
do
	param_IN_summary="${param_IN_summary} -i ${IN_summary[$i]}"
done
# #
python $EXE $param_IN_summary -o $OUT_dir
