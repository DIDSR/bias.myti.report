#!/bin/bash
# # Program to 
# # to run: >> ./run_main_summarize.sh
# # 
# # 
# #
EXE="../src/main_summarize.py"
# # for testing purposes
declare -a IN_paths=("/gpfs_projects/common_data/MIDRC/Release_1c/manifest-1610656454899/MIDRC-RICORD-1C/")
declare -a IN_Repos=("RICORD_1c")
declare -a OUT_summary=("/gpfs_projects/alexis.burgon/OUT/2022_CXR/summary_table_RICORD_1c.json")
# # declare -a IN_paths=("/home/ravi.samala/DATA/temp/open_AI_unzip/" "/gpfs_projects/common_data/TCIA/COVID_19_AR/manifest-1594658036421/COVID-19-AR/" "/home/ravi.samala/DATA/COVID_19_NY_SBU/manifest-1628608914773/COVID-19-NY-SBU/")
# # declare -a IN_Repos=("open_AI" "COVID_19_AR" "COVID_19_NY_SBU")
# # declare -a OUT_summary=("/gpfs_projects/alexis.burgon/OUT/2022_CXR/summary_table__open_AI.tsv" "/gpfs_projects/alexis.burgon/OUT/2022_CXR/summary_table__COVID_19_AR.tsv" "/gpfs_projects/alexis.burgon/OUT/2022_CXR/summary_table__COVID_19_NY_SBU2.tsv")
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
