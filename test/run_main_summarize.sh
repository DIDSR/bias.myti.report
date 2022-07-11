#!/bin/bash
# # Program to 
# # to run: >> ./run_main_summarize.sh
# # 
# # 
# #
EXE="../src/main_summarize.py"
# #
# declare -a IN_paths=("/home/ravi.samala/DATA/temp/open_AI_unzip/" "/home/ravi.samala/DATA/COVID_19_AR/manifest-1594658036421/COVID-19-AR/" "/home/ravi.samala/DATA/COVID_19_NY_SBU/manifest-1628608914773/COVID-19-NY-SBU/")
declare -a IN_paths=("/gpfs_projects/ravi.samala/DATA/MIDRC2/open_RI_unzip/" "/gpfs_projects/ravi.samala/DATA/MIDRC2/open_AI_unzip/" "/home/ravi.samala/DATA/COVID_19_AR/manifest-1594658036421/COVID-19-AR/" "/home/ravi.samala/DATA/COVID_19_NY_SBU/manifest-1628608914773/COVID-19-NY-SBU/")
declare -a IN_Repos=("open_RI" "open_AI" "COVID_19_AR" "COVID_19_NY_SBU")
declare -a OUT_summary=("/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__open_RI.json" "/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__open_AI.json" "/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__COVID_19_AR.tsv" "/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__COVID_19_NY_SBU2.tsv")
# #
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
param_OUT_summary=""
# # create input params
for (( i=0; i<${arraylength}; i++ ));
do
	param_IN_paths="${param_IN_paths} -i ${IN_paths[$i]}"
	param_IN_Repos="${param_IN_Repos} -n ${IN_Repos[$i]}"
	param_OUT_summary="${param_OUT_summary} -o ${OUT_summary[$i]}"
done
# #
python $EXE $param_IN_paths $param_IN_Repos $param_OUT_summary
