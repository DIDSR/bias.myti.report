#!/bin/bash
# # Program to 
# # to run: >> ./run_main_summarize.sh
# # 
# # 
# #
EXE="/gpfs_projects/ravi.samala/projects/2022/summarize_cxr/main_summarize.py"
# #
declare -a IN_paths=("/home/ravi.samala/DATA/COVID_19_AR/manifest-1594658036421/COVID-19-AR/" "/home/ravi.samala/DATA/COVID_19_NY_SBU/manifest-1628608914773/COVID-19-NY-SBU/" "element3")
declare -a IN_Repos=("COVID_19_AR" "COVID_19_NY_SBU" "element3")
# #
# OUT_file="/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__COVID_19_NY_SBU2.tsv"
OUT_file="/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__COVID_19_AR.tsv"
#
#
# #
# #
# # -----------------------------------------------------------------------
# # -----------------------------------------------------------------------
# # -----------------------------------------------------------------------
# #
# # get length of an array
arraylength=${#IN_paths[@]}
# #
param_IN_paths=""
param_IN_Repos=""
# create input params
for (( i=0; i<${arraylength}; i++ ));
do
	param_IN_paths="${param_IN_paths} -i ${IN_paths[$i]}"
	param_IN_Repos="${param_IN_Repos} -n ${IN_Repos[$i]}"
	# # echo "index: $i, value: ${IN_paths[$i]}"
done
# echo $param_IN_paths
# echo $param_IN_Repos
python $EXE $param_IN_paths $param_IN_Repos -o $OUT_file
